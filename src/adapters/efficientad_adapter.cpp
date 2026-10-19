#include "adapters/efficientad_adapter.h"

#include "adapters/feature_utils.h"
#include "compute/gpu_compute.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

namespace anom::model {
namespace {

Result<std::string> bindingFor(
    const std::unordered_map<std::string, std::string>& bindings,
    const std::string& semantic) {
    const auto found = bindings.find(semantic);
    if (found == bindings.end()) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "EfficientAD output semantic is not bound", semantic);
    }
    return found->second;
}

Result<const Tensor*> streamFrom(const TensorMap& outputs, const std::string& name,
                                 const std::string& semantic) {
    const auto found = outputs.find(name);
    if (found == outputs.end()) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Backend did not return EfficientAD stream", name);
    }
    return &found->second;
}

// Normalize an [N, C, H, W] float32 tensor to a single-channel [N, 1, H, W]
// map. A single-channel input passes through unchanged; a multi-channel input
// is collapsed with the per-location L2 norm.
Result<Tensor> singleChannelMap(const Tensor& tensor, const std::string& context) {
    if (tensor.dtype != DataType::Float32 || tensor.shape.dims.size() != 4) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "EfficientAD stream must be a float32 rank-4 tensor",
                             context + " " + tensor.shape.toString());
    }
    if (tensor.shape.dims[1] == 1) return tensor;
    if (tensor.shape.dims[1] < 0) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "EfficientAD stream channel dimension must be static",
                             context + " " + tensor.shape.toString());
    }
    return feature::channelWiseL2(tensor);
}

cv::Mat planeView(const Tensor& singleChannel, int batch) {
    const int height = static_cast<int>(singleChannel.shape.dims[2]);
    const int width = static_cast<int>(singleChannel.shape.dims[3]);
    const std::size_t elements = static_cast<std::size_t>(height) * width;
    return cv::Mat(height, width, CV_32FC1,
                   const_cast<float*>(singleChannel.data<float>() +
                                      static_cast<std::size_t>(batch) * elements));
}

}  // namespace

class EfficientADAdapter::Impl {
public:
    Impl(EfficientADConfig config,
         std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage&) { return {}; }

    Result<void> configureCompute(int device, const std::filesystem::path& directory) {
        if (device < 0) {
            compute_.reset();
            return {};
        }
        const float totalWeight = config_.hardWeight + config_.softWeight;
        if (!std::isfinite(config_.hardWeight) || config_.hardWeight < 0 ||
            !std::isfinite(config_.softWeight) || config_.softWeight < 0 ||
            !std::isfinite(totalWeight) || totalWeight <= 0 ||
            !std::isfinite(config_.gaussianSigma) || config_.gaussianSigma < 0) {
            return Status::error(ErrorCode::InvalidArgument, "Invalid EfficientAD compute parameters");
        }
        if (config_.gaussianSigma > 511.0F) {
            return Status::error(ErrorCode::DeviceUnavailable,
                                 "EfficientAD Gaussian kernel exceeds CUDA support");
        }
        auto loaded = GpuCompute::load(device, directory);
        if (!loaded) return loaded.status();
        const uint64_t operations = (uint64_t{1} << ANOM_COMPUTE_PREPROCESS) |
                                    (uint64_t{1} << ANOM_COMPUTE_CHANNEL_L2) |
                                    (uint64_t{1} << ANOM_COMPUTE_NORMALIZED_BLEND) |
                                    (uint64_t{1} << ANOM_COMPUTE_MAP);
        auto supported = loaded.value()->supportsOperations(operations);
        if (!supported) return supported.status();
        compute_ = std::move(loaded.value());
        return {};
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        auto teacherName = bindingFor(outputBindings_, config_.teacherSemantic);
        if (!teacherName) return teacherName.status();
        auto studentName = bindingFor(outputBindings_, config_.studentSemantic);
        if (!studentName) return studentName.status();
        auto reconName = bindingFor(outputBindings_, config_.reconstructionErrorSemantic);
        if (!reconName) return reconName.status();

        const auto* teacher = signature.findOutput(teacherName.value());
        if (!teacher) return Status::error(ErrorCode::TensorSignatureMismatch,
                                           "EfficientAD teacher output is absent from backend",
                                           teacherName.value());
        const auto* student = signature.findOutput(studentName.value());
        if (!student) return Status::error(ErrorCode::TensorSignatureMismatch,
                                           "EfficientAD student output is absent from backend",
                                           studentName.value());
        const auto* recon = signature.findOutput(reconName.value());
        if (!recon) return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "EfficientAD reconstruction error output is absent from backend",
                                         reconName.value());

        if (teacher->dtype != DataType::Float32 || teacher->shape.dims.size() != 4 ||
            student->dtype != DataType::Float32 || student->shape.dims.size() != 4) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "EfficientAD teacher and student must be float32 rank-4 tensors");
        }
        if (recon->dtype != DataType::Float32 || recon->shape.dims.size() != 4) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "EfficientAD reconstruction error must be a float32 rank-4 tensor",
                                 reconName.value() + " " + recon->shape.toString());
        }
        int64_t staticBatch = -1;
        for (const auto* stream : {teacher, student, recon}) {
            for (const auto dimension : stream->shape.dims) {
                if (dimension == 0 || dimension < -1 || dimension > INT_MAX)
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "EfficientAD stream has an invalid dimension", stream->name);
            }
            const auto batch = stream->shape.dims[0];
            if (batch > 0) {
                if (staticBatch > 0 && staticBatch != batch)
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "EfficientAD stream batches differ", stream->name);
                staticBatch = batch;
            }
        }
        for (std::size_t i = 1; i < 4; ++i) {
            const auto teacherDim = teacher->shape.dims[i];
            const auto studentDim = student->shape.dims[i];
            if (teacherDim > 0 && studentDim > 0 && teacherDim != studentDim) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "EfficientAD teacher and student streams disagree on shape",
                                     teacher->shape.toString() + " vs " + student->shape.toString());
            }
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& inputSize) const {
        auto teacherName = bindingFor(outputBindings_, config_.teacherSemantic);
        if (!teacherName) return teacherName.status();
        auto studentName = bindingFor(outputBindings_, config_.studentSemantic);
        if (!studentName) return studentName.status();
        auto reconName = bindingFor(outputBindings_, config_.reconstructionErrorSemantic);
        if (!reconName) return reconName.status();

        auto teacher = streamFrom(outputs, teacherName.value(), config_.teacherSemantic);
        if (!teacher) return teacher.status();
        auto student = streamFrom(outputs, studentName.value(), config_.studentSemantic);
        if (!student) return student.status();
        auto recon = streamFrom(outputs, reconName.value(), config_.reconstructionErrorSemantic);
        if (!recon) return recon.status();

        if (compute_) return predictGpu(*teacher.value(), *student.value(), *recon.value(), inputSize);

        auto hard = feature::channelWiseL2Difference(*teacher.value(), *student.value());
        if (!hard) return hard.status();
        auto soft = singleChannelMap(*recon.value(), "reconstruction_error");
        if (!soft) return soft.status();

        if (hard.value().shape.dims[0] != soft.value().shape.dims[0]) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "EfficientAD hard and soft streams disagree on batch");
        }
        const int batch = static_cast<int>(hard.value().shape.dims[0]);
        if (batch <= 0) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "EfficientAD stream has an empty batch dimension");
        }

        auto hardResized =
            feature::resizeNchw(hard.value(), inputSize.height, inputSize.width);
        if (!hardResized) return hardResized.status();
        auto softResized =
            feature::resizeNchw(soft.value(), inputSize.height, inputSize.width);
        if (!softResized) return softResized.status();

        const float totalWeight = config_.hardWeight + config_.softWeight;
        const float hardAlpha = config_.hardWeight / totalWeight;
        const float softAlpha = config_.softWeight / totalWeight;

        RawPredictionBatch result(static_cast<std::size_t>(batch));
        for (int n = 0; n < batch; ++n) {
            cv::Mat hardMap = planeView(hardResized.value(), n);
            cv::Mat softMap = planeView(softResized.value(), n);
            feature::normalizeMinMax(hardMap);
            feature::normalizeMinMax(softMap);

            cv::Mat combined(inputSize, CV_32FC1);
            cv::addWeighted(hardMap, hardAlpha, softMap, softAlpha, 0.0, combined);
            if (config_.gaussianSigma > 0.0F) {
                cv::GaussianBlur(combined, combined, cv::Size(0, 0),
                                 config_.gaussianSigma, config_.gaussianSigma);
            }

            double minimum = 0.0, maximum = 0.0;
            cv::minMaxLoc(combined, &minimum, &maximum);
            result[static_cast<std::size_t>(n)].score = static_cast<float>(maximum);
            result[static_cast<std::size_t>(n)].anomalyMap = combined;
        }
        return result;
    }

private:
    Result<RawPredictionBatch> predictGpu(const Tensor& teacher, const Tensor& student,
                                         const Tensor& reconstruction, const cv::Size& inputSize) const {
        try {
            if (inputSize.height <= 0 || inputSize.width <= 0 ||
                reconstruction.dtype != DataType::Float32 || reconstruction.shape.dims.size() != 4) {
                throw std::invalid_argument("Invalid EfficientAD reconstruction or output shape");
            }
            for (auto dimension : reconstruction.shape.dims) {
                if (dimension <= 0 || dimension > INT_MAX)
                    throw std::invalid_argument("Invalid EfficientAD reconstruction dimension");
            }
            const auto reconstructionCount = reconstruction.shape.elementCount();
            if (!reconstructionCount || reconstructionCount.value() > SIZE_MAX / sizeof(float) ||
                reconstruction.byteSize() != reconstructionCount.value() * sizeof(float))
                throw std::invalid_argument("EfficientAD reconstruction buffer size differs from shape");
            const auto hard = compute_->channelL2(teacher, &student);
            // A one-channel reconstruction stream already contains error scores.
            // Preserve its sign; only multi-channel streams need an L2 reduction.
            const auto soft = reconstruction.shape.dims[1] == 1
                                  ? compute_->upload(reconstruction)
                                  : compute_->channelL2(reconstruction);
            const int batch = static_cast<int>(hard.shape.dims[0]);
            if (soft.shape.dims[0] != batch)
                throw std::invalid_argument("EfficientAD hard and soft streams disagree on batch");
            const auto hardResized = compute_->maps(hard, batch, static_cast<int>(hard.shape.dims[2]),
                                                    static_cast<int>(hard.shape.dims[3]), inputSize, 0.0F);
            const auto softResized = compute_->maps(soft, batch, static_cast<int>(soft.shape.dims[2]),
                                                    static_cast<int>(soft.shape.dims[3]), inputSize, 0.0F);
            auto combined = compute_->normalizedBlend(hardResized, softResized,
                                                       config_.hardWeight, config_.softWeight);
            if (config_.gaussianSigma > 0.0F) {
                combined = compute_->maps(combined, batch, inputSize.height, inputSize.width,
                                           inputSize, config_.gaussianSigma);
            }
            const auto host = compute_->download(combined);
            const size_t pixels = static_cast<size_t>(inputSize.height) * inputSize.width;
            RawPredictionBatch result(static_cast<size_t>(batch));
            for (int n = 0; n < batch; ++n) {
                auto map = cv::Mat(inputSize, CV_32FC1,
                                   const_cast<float*>(host.data<float>()) + static_cast<size_t>(n) * pixels).clone();
                double maximum = 0.0;
                // The API returns a host map; derive its final, post-blur score here.
                cv::minMaxLoc(map, nullptr, &maximum);
                result[static_cast<size_t>(n)].score = static_cast<float>(maximum);
                result[static_cast<size_t>(n)].anomalyMap = std::move(map);
            }
            return result;
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::CudaFailure, "GPU EfficientAD prediction failed", error.what());
        }
    }

    EfficientADConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    std::shared_ptr<GpuCompute> compute_;
};

EfficientADAdapter::EfficientADAdapter(
    EfficientADConfig config,
    std::unordered_map<std::string, std::string> outputBindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(outputBindings))) {}
EfficientADAdapter::~EfficientADAdapter() = default;
Result<void> EfficientADAdapter::loadAssets(const ModelPackage& package) {
    return impl_->loadAssets(package);
}
Result<void> EfficientADAdapter::configureCompute(int device, const std::filesystem::path& directory) {
    return impl_->configureCompute(device, directory);
}
Result<void> EfficientADAdapter::validateSignature(const TensorSignature& signature) const {
    return impl_->validateSignature(signature);
}
Result<RawPredictionBatch> EfficientADAdapter::predict(const TensorMap& outputs,
                                                       const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
