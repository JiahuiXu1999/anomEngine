#include "pipeline/efficientad.h"

#include "adapters/feature_utils.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'N', 'E', 'F', 'F', 'A', 'D', '\0'};
constexpr std::uint32_t kStatisticsVersion = 1;

template <typename T>
bool writeValue(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

Result<const Tensor*> findStream(const TensorMap& outputs, const std::string& name) {
    const auto found = outputs.find(name);
    if (found == outputs.end()) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Backend did not return EfficientAD stream", name);
    }
    return &found->second;
}

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

void normalize01(cv::Mat& map) {
    double minimum = 0.0, maximum = 0.0;
    cv::minMaxLoc(map, &minimum, &maximum);
    if (maximum > minimum) {
        map = (map - minimum) / static_cast<float>(maximum - minimum);
    } else {
        map = cv::Scalar(0.0F);
    }
}

}  // namespace

EfficientADPipeline::EfficientADPipeline(EfficientADPipelineConfig config)
    : config_(std::move(config)),
      imageMin_(std::numeric_limits<double>::infinity()),
      imageMax_(-std::numeric_limits<double>::infinity()),
      pixelMin_(std::numeric_limits<double>::infinity()),
      pixelMax_(-std::numeric_limits<double>::infinity()) {}

Result<void> EfficientADPipeline::validateConfig() const {
    if (!std::isfinite(config_.hardWeight) || config_.hardWeight < 0.0F ||
        !std::isfinite(config_.softWeight) || config_.softWeight < 0.0F ||
        config_.hardWeight + config_.softWeight <= 0.0F) {
        return Status::error(ErrorCode::InvalidArgument,
                             "EfficientAD pipeline hard and soft weights must be finite, "
                             "non-negative, and not both zero");
    }
    if (!std::isfinite(config_.gaussianSigma) || config_.gaussianSigma < 0.0F) {
        return Status::error(ErrorCode::InvalidArgument,
                             "EfficientAD pipeline gaussian sigma must be finite and non-negative");
    }
    return {};
}

Result<void> EfficientADPipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const EfficientADConfig& algorithmConfig,
    const cv::Size& inputSize) {
    auto valid = validateConfig();
    if (!valid) return valid.status();

    auto teacherName = bindingFor(outputBindings, algorithmConfig.teacherSemantic);
    if (!teacherName) return teacherName.status();
    auto studentName = bindingFor(outputBindings, algorithmConfig.studentSemantic);
    if (!studentName) return studentName.status();
    auto reconName = bindingFor(outputBindings, algorithmConfig.reconstructionErrorSemantic);
    if (!reconName) return reconName.status();

    auto teacher = findStream(outputs, teacherName.value());
    if (!teacher) return teacher.status();
    auto student = findStream(outputs, studentName.value());
    if (!student) return student.status();
    auto recon = findStream(outputs, reconName.value());
    if (!recon) return recon.status();

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

    for (int n = 0; n < batch; ++n) {
        const int height = inputSize.height;
        const int width = inputSize.width;
        const std::size_t elements = static_cast<std::size_t>(height) * width;
        cv::Mat hardMap(height, width, CV_32FC1,
                        const_cast<float*>(hardResized.value().data<float>() +
                                           static_cast<std::size_t>(n) * elements));
        cv::Mat softMap(height, width, CV_32FC1,
                        const_cast<float*>(softResized.value().data<float>() +
                                           static_cast<std::size_t>(n) * elements));
        normalize01(hardMap);
        normalize01(softMap);

        cv::Mat combined(height, width, CV_32FC1);
        cv::addWeighted(hardMap, hardAlpha, softMap, softAlpha, 0.0, combined);
        if (config_.gaussianSigma > 0.0F) {
            cv::GaussianBlur(combined, combined, cv::Size(0, 0),
                             config_.gaussianSigma, config_.gaussianSigma);
        }
        accumulate(combined);
    }
    return {};
}

Result<void> EfficientADPipeline::addAnomalyMap(const cv::Mat& map) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    if (map.empty() || map.type() != CV_32FC1) {
        return Status::error(ErrorCode::InvalidArgument,
                             "EfficientAD anomaly map must be a non-empty CV_32FC1 matrix");
    }
    accumulate(map);
    return {};
}

void EfficientADPipeline::accumulate(const cv::Mat& combinedMap) {
    double minimum = 0.0, maximum = 0.0;
    cv::minMaxLoc(combinedMap, &minimum, &maximum);
    imageMin_ = std::min(imageMin_, maximum);
    imageMax_ = std::max(imageMax_, maximum);
    pixelMin_ = std::min(pixelMin_, minimum);
    pixelMax_ = std::max(pixelMax_, maximum);
    ++sampleCount_;
}

Result<void> EfficientADPipeline::saveStatistics(
    const std::filesystem::path& path) const {
    if (sampleCount_ == 0) {
        return Status::error(ErrorCode::NotInitialized,
                             "EfficientAD pipeline contains no training samples");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to create EfficientAD statistics artifact",
                             path.string());
    }
    stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    if (!writeValue(stream, kStatisticsVersion) ||
        !writeValue(stream, static_cast<float>(imageMin_)) ||
        !writeValue(stream, static_cast<float>(imageMax_)) ||
        !writeValue(stream, static_cast<float>(pixelMin_)) ||
        !writeValue(stream, static_cast<float>(pixelMax_))) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write EfficientAD statistics artifact",
                             path.string());
    }
    return {};
}

void EfficientADPipeline::clear() {
    sampleCount_ = 0;
    imageMin_ = std::numeric_limits<double>::infinity();
    imageMax_ = -std::numeric_limits<double>::infinity();
    pixelMin_ = std::numeric_limits<double>::infinity();
    pixelMax_ = -std::numeric_limits<double>::infinity();
}

}  // namespace anom::model
