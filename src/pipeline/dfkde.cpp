#include "pipeline/dfkde.h"

#include "adapters/feature_utils.h"

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'N', 'D', 'F', 'K', 'D', 'E', '\0'};
constexpr std::uint32_t kStatisticsVersion = 1;

template <typename T>
bool writeValue(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

void writeFloats(std::ofstream& stream, const float* data, std::size_t count) {
    stream.write(reinterpret_cast<const char*>(data),
                 static_cast<std::streamsize>(count * sizeof(float)));
}

}  // namespace

DfkdePipeline::DfkdePipeline(DfkdePipelineConfig config)
    : config_(std::move(config)) {}

Result<void> DfkdePipeline::validateConfig() const {
    if (config_.embeddingDimension <= 0 || config_.nComponents <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "DFKDE embedding dimension and component count must be positive");
    }
    if (!std::isfinite(config_.kernelSigma) || config_.kernelSigma <= 0.0F) {
        return Status::error(ErrorCode::InvalidArgument,
                             "DFKDE kernel sigma must be finite and positive");
    }
    return {};
}

Result<void> DfkdePipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const DFKDEConfig& algorithmConfig) {
    if (algorithmConfig.embeddingDimension != config_.embeddingDimension) {
        return Status::error(ErrorCode::InvalidArgument,
                             "DFKDE pipeline and manifest embedding dimensions differ");
    }
    auto embedding = feature::aggregateFeaturePyramid(
        outputs, outputBindings, algorithmConfig.featureLayers, false, 1, 1, 0);
    if (!embedding) return embedding.status();
    return addEmbedding(embedding.value());
}

Result<void> DfkdePipeline::addEmbedding(const Tensor& embedding) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    auto view = feature::viewNchwFloat(embedding, "DFKDE training embedding");
    if (!view) return view.status();
    if (view.value().channels != config_.embeddingDimension) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "DFKDE training embedding has the wrong channel count",
                             embedding.shape.toString());
    }
    const auto flattened = feature::flattenPatches(embedding);
    const std::size_t count = static_cast<std::size_t>(view.value().batch) *
                              view.value().height * view.value().width;
    return addFeatures(flattened, count);
}

Result<void> DfkdePipeline::addFeatures(const std::vector<float>& features,
                                        std::size_t vectorCount) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    if (vectorCount == 0 || vectorCount > std::numeric_limits<std::size_t>::max() / dimension ||
        features.size() != vectorCount * dimension) {
        return Status::error(ErrorCode::InvalidArgument,
                             "DFKDE feature matrix has an invalid size");
    }
    for (const float value : features) {
        if (!std::isfinite(value)) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "DFKDE feature matrix contains non-finite values");
        }
    }
    if (features_.size() > std::numeric_limits<std::size_t>::max() - features.size()) {
        return Status::error(ErrorCode::OutOfMemory,
                             "DFKDE feature matrix size overflows size_t");
    }
    features_.insert(features_.end(), features.begin(), features.end());
    return {};
}

Result<void> DfkdePipeline::saveStatistics(const std::filesystem::path& path) const {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    const int dimension = config_.embeddingDimension;
    const std::size_t vectorCount = features_.size() / static_cast<std::size_t>(dimension);
    if (vectorCount == 0) {
        return Status::error(ErrorCode::NotInitialized,
                             "DFKDE pipeline contains no training features");
    }
    const int components = std::min(config_.nComponents,
                                    std::min(dimension, static_cast<int>(vectorCount)));
    if (components <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "DFKDE component count must be positive");
    }

    cv::Mat data(static_cast<int>(vectorCount), dimension, CV_32FC1,
                 const_cast<float*>(features_.data()));
    cv::Mat mean, eigenvectors;
    cv::PCACompute(data, mean, eigenvectors, components);

    cv::Mat meanTiled;
    cv::repeat(mean, static_cast<int>(vectorCount), 1, meanTiled);
    cv::Mat centered = data - meanTiled;
    cv::Mat gallery = centered * eigenvectors;  // N x components

    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to create DFKDE statistics artifact", path.string());
    }
    stream.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    if (!writeValue(stream, kStatisticsVersion) ||
        !writeValue(stream, static_cast<std::uint32_t>(dimension)) ||
        !writeValue(stream, static_cast<std::uint32_t>(components)) ||
        !writeValue(stream, static_cast<std::uint32_t>(vectorCount))) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write DFKDE statistics header", path.string());
    }
    writeFloats(stream, mean.ptr<float>(0), static_cast<std::size_t>(dimension));
    writeFloats(stream, eigenvectors.ptr<float>(0),
                static_cast<std::size_t>(dimension) * components);
    writeFloats(stream, gallery.ptr<float>(0),
                static_cast<std::size_t>(vectorCount) * components);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write DFKDE statistics payload", path.string());
    }
    return {};
}

void DfkdePipeline::clear() noexcept { features_.clear(); }

std::size_t DfkdePipeline::featureCount() const noexcept {
    return config_.embeddingDimension > 0
               ? features_.size() / static_cast<std::size_t>(config_.embeddingDimension)
               : 0;
}

}  // namespace anom::model
