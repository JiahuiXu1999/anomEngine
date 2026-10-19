#include "adapters/dfkde_adapter.h"

#include "adapters/feature_utils.h"
#include "compute/gpu_compute.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'N', 'D', 'F', 'K', 'D', 'E', '\0'};
constexpr std::uint32_t kStatisticsVersion = 1;

template <typename T>
bool readScalar(const std::vector<char>& data, std::size_t& offset, T& value) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) return false;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

Result<std::vector<char>> readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return Status::error(ErrorCode::IoError, "Unable to open DFKDE artifact", path.string());
    const auto size = stream.tellg();
    if (size <= 0) return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE artifact is empty", path.string());
    std::vector<char> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        return Status::error(ErrorCode::IoError, "Unable to read complete DFKDE artifact", path.string());
    }
    return bytes;
}

}  // namespace

class DfkdeAdapter::Impl {
public:
    Impl(DFKDEConfig config, std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage& package) {
        auto path = package.resolveArtifact(config_.statisticsFile);
        if (!path) return path.status();
        auto bytes = readFile(path.value());
        if (!bytes) return bytes.status();

        std::size_t offset = 0;
        std::array<char, 8> magic{};
        if (bytes.value().size() < magic.size()) {
            return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE statistics header is truncated");
        }
        std::memcpy(magic.data(), bytes.value().data(), magic.size());
        offset += magic.size();
        std::uint32_t version = 0, dimension = 0, components = 0, galleryCount = 0;
        if (magic != kMagic ||
            !readScalar(bytes.value(), offset, version) ||
            !readScalar(bytes.value(), offset, dimension) ||
            !readScalar(bytes.value(), offset, components) ||
            !readScalar(bytes.value(), offset, galleryCount)) {
            return Status::error(ErrorCode::ArtifactCorrupt, "Invalid DFKDE statistics header");
        }
        if (version != kStatisticsVersion ||
            dimension != static_cast<std::uint32_t>(config_.embeddingDimension)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "DFKDE statistics metadata does not match manifest");
        }
        if (dimension == 0 || dimension > INT_MAX || components == 0 ||
            components > dimension || galleryCount == 0 || galleryCount > INT_MAX) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "DFKDE statistics has invalid dimensions");
        }
        const std::size_t meanCount = dimension;
        if (size_t(dimension) > SIZE_MAX / components || size_t(galleryCount) > SIZE_MAX / components) {
            return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE statistics size overflows size_t");
        }
        const std::size_t eigenCount = static_cast<std::size_t>(dimension) * components;
        const std::size_t gallerySize = static_cast<std::size_t>(galleryCount) * components;
        if (eigenCount > SIZE_MAX - meanCount || gallerySize > SIZE_MAX - meanCount - eigenCount) {
            return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE statistics size overflows size_t");
        }
        const std::size_t totalFloats = meanCount + eigenCount + gallerySize;
        if (totalFloats > SIZE_MAX / sizeof(float)) {
            return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE statistics byte size overflows size_t");
        }
        if (bytes.value().size() - offset != totalFloats * sizeof(float)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "DFKDE statistics payload size does not match header");
        }
        std::vector<float> statistics(totalFloats);
        std::memcpy(statistics.data(), bytes.value().data() + offset, totalFloats * sizeof(float));
        for (const float value : statistics) {
            if (!std::isfinite(value)) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "DFKDE statistics contain non-finite values");
            }
        }
        mean_ = std::move(statistics);
        dimension_ = static_cast<int>(dimension);
        components_ = static_cast<int>(components);
        galleryCount_ = static_cast<int>(galleryCount);
        deviceMean_ = {};
        deviceBasis_ = {};
        deviceGallery_ = {};
        compute_.reset();
        loaded_ = true;
        return {};
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        std::set<std::string> semantics;
        std::set<std::string> tensorNames;
        std::int64_t staticBatch = -1;
        std::int64_t staticChannels = 0;
        bool allChannelsStatic = true;
        for (const auto& semantic : config_.featureLayers) {
            if (!semantics.insert(semantic).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature semantic is duplicated", semantic);
            }
            const auto binding = outputBindings_.find(semantic);
            if (binding == outputBindings_.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature semantic is not bound", semantic);
            }
            if (!tensorNames.insert(binding->second).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature layers must bind distinct tensors",
                                     binding->second);
            }
            const auto* spec = signature.findOutput(binding->second);
            if (!spec || spec->dtype != DataType::Float32 || spec->shape.dims.size() != 4) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature must be a float32 NCHW output", binding->second);
            }
            for (const auto value : spec->shape.dims) {
                if (value == 0 || value < -1 || value > INT_MAX) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "DFKDE feature has an invalid dimension", binding->second);
                }
            }
            if (spec->shape.dims[0] > 0) {
                if (staticBatch > 0 && staticBatch != spec->shape.dims[0]) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "DFKDE feature batches differ", binding->second);
                }
                staticBatch = spec->shape.dims[0];
            }
            if (spec->shape.dims[1] > 0) {
                staticChannels += spec->shape.dims[1];
            } else {
                allChannelsStatic = false;
            }
        }
        if (allChannelsStatic && staticChannels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "DFKDE feature channels do not match the embedding dimension",
                                 std::to_string(staticChannels) + " != " +
                                     std::to_string(config_.embeddingDimension));
        }
        return {};
    }

    Result<void> configureCompute(int device, const std::filesystem::path& directory) {
        if (device < 0) {
            deviceMean_ = {};
            deviceBasis_ = {};
            deviceGallery_ = {};
            compute_.reset();
            return {};
        }
        if (!loaded_) return Status::error(ErrorCode::NotInitialized, "DFKDE statistics are not loaded");
        if (!std::isfinite(config_.gaussianSigma) || config_.gaussianSigma < 0.0F ||
            config_.gaussianSigma > 511.0F) {
            return Status::error(ErrorCode::DeviceUnavailable, "DFKDE Gaussian kernel exceeds CUDA support");
        }
        if (!std::isfinite(config_.kernelSigma) || config_.kernelSigma <= 0.0F ||
            !std::isfinite(1.0F / (2.0F * config_.kernelSigma * config_.kernelSigma))) {
            return Status::error(ErrorCode::DeviceUnavailable, "DFKDE inverse bandwidth is non-finite");
        }
        auto loaded = GpuCompute::load(device, directory);
        if (!loaded) return loaded.status();
        const uint64_t operations = (uint64_t(1) << ANOM_COMPUTE_PREPROCESS) |
                                    (uint64_t(1) << ANOM_COMPUTE_AGGREGATE) |
                                    (uint64_t(1) << ANOM_COMPUTE_DFKDE) |
                                    (uint64_t(1) << ANOM_COMPUTE_MAP);
        auto supported = loaded.value()->supportsOperations(operations);
        if (!supported) return supported.status();
        try {
            const size_t basisCount = size_t(dimension_) * components_;
            Tensor mean, basis, gallery;
            mean.shape.dims = {dimension_};
            mean.setExternalView(mean_.data(), size_t(dimension_) * sizeof(float));
            basis.shape.dims = {dimension_, components_};
            basis.setExternalView(mean_.data() + dimension_, basisCount * sizeof(float));
            gallery.shape.dims = {galleryCount_, components_};
            gallery.setExternalView(mean_.data() + dimension_ + basisCount,
                                     size_t(galleryCount_) * components_ * sizeof(float));
            auto deviceMean = loaded.value()->upload(mean);
            auto deviceBasis = loaded.value()->upload(basis);
            auto deviceGallery = loaded.value()->upload(gallery);
            compute_ = std::move(loaded.value());
            deviceMean_ = std::move(deviceMean);
            deviceBasis_ = std::move(deviceBasis);
            deviceGallery_ = std::move(deviceGallery);
            return {};
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::CudaFailure, "Unable to upload DFKDE statistics", error.what());
        }
    }

    Result<RawPredictionBatch> predictGpu(const TensorMap& outputs, const cv::Size& inputSize) const {
        try {
            auto patches = compute_->aggregateFeatures(outputs, outputBindings_, config_.featureLayers);
            if (patches.shape.dims[3] != dimension_) {
                return Status::error(ErrorCode::TensorShapeMismatch,
                                     "DFKDE embedding dimension does not match statistics",
                                     patches.shape.toString());
            }
            const int batch = static_cast<int>(patches.shape.dims[0]);
            const int height = static_cast<int>(patches.shape.dims[1]);
            const int width = static_cast<int>(patches.shape.dims[2]);
            auto scores = compute_->dfkdeScores(patches, deviceMean_, deviceBasis_, deviceGallery_,
                                                config_.kernelSigma);
            const auto maps = compute_->download(
                compute_->maps(scores, batch, height, width, inputSize, config_.gaussianSigma));
            RawPredictionBatch result(static_cast<size_t>(batch));
            const size_t pixels = size_t(inputSize.height) * inputSize.width;
            for (int n = 0; n < batch; ++n) {
                auto map = cv::Mat(inputSize.height, inputSize.width, CV_32FC1,
                                   const_cast<float*>(maps.data<float>()) + size_t(n) * pixels,
                                   size_t(inputSize.width) * sizeof(float)).clone();
                double maximum = 0;
                // DFKDE scores the final resized and smoothed map. The public
                // output requires this map on CPU, so the maximum needs no copy.
                cv::minMaxLoc(map, nullptr, &maximum);
                result[size_t(n)].score = static_cast<float>(maximum);
                result[size_t(n)].anomalyMap = std::move(map);
            }
            return result;
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::CudaFailure, "GPU DFKDE prediction failed", error.what());
        }
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& inputSize) const {
        if (!loaded_) return Status::error(ErrorCode::NotInitialized, "DFKDE statistics are not loaded");
        if (compute_) return predictGpu(outputs, inputSize);
        auto embedding = feature::aggregateFeaturePyramid(
            outputs, outputBindings_, config_.featureLayers, false, 1, 1, 0);
        if (!embedding) return embedding.status();
        auto view = feature::viewNchwFloat(embedding.value(), "DFKDE embedding");
        if (!view) return view.status();
        if (view.value().channels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "DFKDE embedding dimension does not match statistics",
                                 embedding.value().shape.toString());
        }

        const std::vector<float> patches = feature::flattenPatches(embedding.value());
        const int locations = view.value().height * view.value().width;
        const int batch = view.value().batch;
        const float inv2s2 = 1.0F / (2.0F * config_.kernelSigma * config_.kernelSigma);

        RawPredictionBatch result(static_cast<std::size_t>(batch));
        std::vector<double> sqDists(static_cast<std::size_t>(galleryCount_));
        std::vector<float> scores(static_cast<std::size_t>(locations));
        for (int n = 0; n < batch; ++n) {
            const float* batchPatches = patches.data() +
                static_cast<std::size_t>(n) * locations * dimension_;
            for (int location = 0; location < locations; ++location) {
                const float* patch = batchPatches + static_cast<std::size_t>(location) * dimension_;
                // Project with the stored PCA basis.
                std::vector<float> projected(static_cast<std::size_t>(components_));
                for (int k = 0; k < components_; ++k) {
                    double sum = 0.0;
                    for (int d = 0; d < dimension_; ++d) {
                        sum += static_cast<double>(patch[d] - mean_[d]) *
                               mean_[static_cast<std::size_t>(dimension_) +
                                     static_cast<std::size_t>(d) * components_ + k];
                    }
                    projected[static_cast<std::size_t>(k)] = static_cast<float>(sum);
                }
                double minSq = std::numeric_limits<double>::infinity();
                for (int g = 0; g < galleryCount_; ++g) {
                    const float* gallery = mean_.data() +
                        static_cast<std::size_t>(dimension_) +
                        static_cast<std::size_t>(dimension_) * components_ +
                        static_cast<std::size_t>(g) * components_;
                    double sq = 0.0;
                    for (int k = 0; k < components_; ++k) {
                        const double diff =
                            static_cast<double>(projected[static_cast<std::size_t>(k)]) - gallery[k];
                        sq += diff * diff;
                    }
                    sqDists[static_cast<std::size_t>(g)] = sq;
                    if (sq < minSq) minSq = sq;
                }
                double logSum = 0.0;
                for (int g = 0; g < galleryCount_; ++g) {
                    logSum += std::exp(-(sqDists[static_cast<std::size_t>(g)] - minSq) * inv2s2);
                }
                scores[static_cast<std::size_t>(location)] =
                    static_cast<float>(minSq * inv2s2 - std::log(logSum));
            }
            auto map = feature::makeAnomalyMap(scores.data(), view.value().height,
                                               view.value().width, inputSize,
                                               config_.gaussianSigma);
            double minimum = 0.0, maximum = 0.0;
            cv::minMaxLoc(map, &minimum, &maximum);
            result[static_cast<std::size_t>(n)].score = static_cast<float>(maximum);
            result[static_cast<std::size_t>(n)].anomalyMap = std::move(map);
        }
        return result;
    }

private:
    DFKDEConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    // Single contiguous buffer: mean [D], eigenvectors [D*K], gallery [N*K].
    std::vector<float> mean_;
    int dimension_{0};
    int components_{0};
    int galleryCount_{0};
    bool loaded_{false};
    std::shared_ptr<GpuCompute> compute_;
    Tensor deviceMean_, deviceBasis_, deviceGallery_;
};

DfkdeAdapter::DfkdeAdapter(DFKDEConfig config,
                           std::unordered_map<std::string, std::string> outputBindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(outputBindings))) {}
DfkdeAdapter::~DfkdeAdapter() = default;
Result<void> DfkdeAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<void> DfkdeAdapter::configureCompute(int device, const std::filesystem::path& directory) {
    return impl_->configureCompute(device, directory);
}
Result<void> DfkdeAdapter::validateSignature(const TensorSignature& signature) const {
    return impl_->validateSignature(signature);
}
Result<RawPredictionBatch> DfkdeAdapter::predict(const TensorMap& outputs,
                                                 const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
