#include "pipeline/padim.h"

#include "adapters/feature_utils.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'N', 'P', 'A', 'D', 'I', 'M', '\0'};
constexpr std::uint32_t kStatisticsVersion = 1;
constexpr std::array<char, 8> kCheckpointMagic{'A', 'N', 'P', 'D', 'C', 'K', 'P', '\0'};
constexpr std::uint32_t kCheckpointVersion = 1;

template <typename T>
bool writeValue(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

template <typename T>
bool readValue(std::ifstream& stream, T& value) {
    stream.read(reinterpret_cast<char*>(&value), sizeof(T));
    return static_cast<bool>(stream);
}

}  // namespace

PadimPipeline::PadimPipeline(PadimPipelineConfig config)
    : config_(std::move(config)) {}

Result<void> PadimPipeline::validateConfig() const {
    if (config_.embeddingDimension <= 0 || config_.featureHeight <= 0 ||
        config_.featureWidth <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PaDiM pipeline dimensions must be positive");
    }
    if (!std::isfinite(config_.covarianceRegularization) ||
        config_.covarianceRegularization <= 0.0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PaDiM covariance regularization must be finite and positive");
    }
    if (config_.channelIndices.size() !=
        static_cast<std::size_t>(config_.embeddingDimension)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PaDiM channel index count must equal embedding dimension");
    }
    std::set<std::int32_t> unique;
    for (const auto index : config_.channelIndices) {
        if (index < 0 || !unique.insert(index).second) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "PaDiM channel indices must be unique and non-negative");
        }
    }
    return {};
}

Result<void> PadimPipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const PadimConfig& algorithmConfig) {
    if (algorithmConfig.embeddingDimension != config_.embeddingDimension ||
        algorithmConfig.featureHeight != config_.featureHeight ||
        algorithmConfig.featureWidth != config_.featureWidth) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PaDiM pipeline and manifest dimensions differ");
    }
    auto embedding = feature::aggregateFeaturePyramid(
        outputs, outputBindings, algorithmConfig.featureLayers, false, 1, 1, 0);
    if (!embedding) return embedding.status();
    return addEmbedding(embedding.value());
}

Result<void> PadimPipeline::addEmbedding(const Tensor& fullEmbedding) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    auto selected = feature::selectChannels(fullEmbedding, config_.channelIndices);
    if (!selected) return selected.status();
    auto view = feature::viewNchwFloat(selected.value(), "PaDiM training embedding");
    if (!view) return view.status();
    if (view.value().channels != config_.embeddingDimension ||
        view.value().height != config_.featureHeight ||
        view.value().width != config_.featureWidth) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "PaDiM training embedding shape does not match configuration",
                             selected.value().shape.toString());
    }

    const std::size_t locations = static_cast<std::size_t>(config_.featureHeight) *
                                  config_.featureWidth;
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    if (locations > std::numeric_limits<std::size_t>::max() / dimension ||
        locations * dimension > std::numeric_limits<std::size_t>::max() / dimension) {
        return Status::error(ErrorCode::OutOfMemory,
                             "PaDiM statistics dimensions overflow size_t");
    }
    if (sums_.empty()) {
        sums_.assign(locations * dimension, 0.0);
        secondMoments_.assign(locations * dimension * dimension, 0.0);
    }
    if (sampleCount_ > std::numeric_limits<std::size_t>::max() -
                           static_cast<std::size_t>(view.value().batch)) {
        return Status::error(ErrorCode::OutOfMemory,
                             "PaDiM sample count overflows size_t");
    }

    const std::vector<float> patches = feature::flattenPatches(selected.value());
    for (int batch = 0; batch < view.value().batch; ++batch) {
        for (std::size_t location = 0; location < locations; ++location) {
            const float* patch = patches.data() +
                (static_cast<std::size_t>(batch) * locations + location) * dimension;
            double* sum = sums_.data() + location * dimension;
            double* moment = secondMoments_.data() + location * dimension * dimension;
            for (std::size_t row = 0; row < dimension; ++row) {
                if (!std::isfinite(patch[row])) {
                    return Status::error(ErrorCode::InvalidArgument,
                                         "PaDiM training embedding contains non-finite values");
                }
                sum[row] += patch[row];
                for (std::size_t column = 0; column < dimension; ++column) {
                    moment[row * dimension + column] +=
                        static_cast<double>(patch[row]) * patch[column];
                }
            }
        }
    }
    sampleCount_ += static_cast<std::size_t>(view.value().batch);
    return {};
}

Result<void> PadimPipeline::saveArtifacts(
    const std::filesystem::path& statisticsPath,
    const std::filesystem::path& channelIndicesPath) const {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    if (sampleCount_ == 0 || sums_.empty() || secondMoments_.empty()) {
        return Status::error(ErrorCode::NotInitialized,
                             "PaDiM pipeline contains no training samples");
    }

    const std::size_t locations = static_cast<std::size_t>(config_.featureHeight) *
                                  config_.featureWidth;
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    std::vector<float> means(locations * dimension);
    std::vector<float> precisions(locations * dimension * dimension);
    const double inverseSamples = 1.0 / static_cast<double>(sampleCount_);
    for (std::size_t location = 0; location < locations; ++location) {
        const double* sum = sums_.data() + location * dimension;
        const double* moment = secondMoments_.data() + location * dimension * dimension;
        cv::Mat covariance(config_.embeddingDimension, config_.embeddingDimension, CV_64FC1);
        for (std::size_t row = 0; row < dimension; ++row) {
            const double mean = sum[row] * inverseSamples;
            means[location * dimension + row] = static_cast<float>(mean);
            for (std::size_t column = 0; column < dimension; ++column) {
                double value = moment[row * dimension + column] * inverseSamples -
                               mean * (sum[column] * inverseSamples);
                if (row == column) value += config_.covarianceRegularization;
                covariance.at<double>(static_cast<int>(row), static_cast<int>(column)) = value;
            }
        }
        cv::Mat precision;
        if (cv::invert(covariance, precision, cv::DECOMP_CHOLESKY) == 0.0) {
            return Status::error(ErrorCode::AdapterFailure,
                                 "PaDiM covariance matrix is not positive definite",
                                 "location " + std::to_string(location));
        }
        float* target = precisions.data() + location * dimension * dimension;
        for (std::size_t row = 0; row < dimension; ++row) {
            for (std::size_t column = 0; column < dimension; ++column) {
                const double value = precision.at<double>(static_cast<int>(row),
                                                          static_cast<int>(column));
                if (!std::isfinite(value) ||
                    std::abs(value) > std::numeric_limits<float>::max()) {
                    return Status::error(ErrorCode::AdapterFailure,
                                         "PaDiM precision matrix cannot be represented as float32",
                                         "location " + std::to_string(location));
                }
                target[row * dimension + column] = static_cast<float>(value);
            }
        }
    }

    std::ofstream indices(channelIndicesPath, std::ios::binary | std::ios::trunc);
    if (!indices) {
        return Status::error(ErrorCode::IoError,
                             "Unable to create PaDiM channel index artifact",
                             channelIndicesPath.string());
    }
    indices.write(reinterpret_cast<const char*>(config_.channelIndices.data()),
                  static_cast<std::streamsize>(config_.channelIndices.size() *
                                               sizeof(std::int32_t)));
    if (!indices) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PaDiM channel index artifact",
                             channelIndicesPath.string());
    }

    std::ofstream statistics(statisticsPath, std::ios::binary | std::ios::trunc);
    if (!statistics) {
        return Status::error(ErrorCode::IoError,
                             "Unable to create PaDiM statistics artifact",
                             statisticsPath.string());
    }
    statistics.write(kMagic.data(), static_cast<std::streamsize>(kMagic.size()));
    const auto height = static_cast<std::uint32_t>(config_.featureHeight);
    const auto width = static_cast<std::uint32_t>(config_.featureWidth);
    const auto dimension32 = static_cast<std::uint32_t>(config_.embeddingDimension);
    if (!writeValue(statistics, kStatisticsVersion) || !writeValue(statistics, height) ||
        !writeValue(statistics, width) || !writeValue(statistics, dimension32)) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PaDiM statistics header",
                             statisticsPath.string());
    }
    statistics.write(reinterpret_cast<const char*>(means.data()),
                     static_cast<std::streamsize>(means.size() * sizeof(float)));
    statistics.write(reinterpret_cast<const char*>(precisions.data()),
                     static_cast<std::streamsize>(precisions.size() * sizeof(float)));
    if (!statistics) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PaDiM statistics artifact",
                             statisticsPath.string());
    }
    return {};
}

Result<void> PadimPipeline::saveCheckpoint(
    const std::filesystem::path& path) const {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    if (sampleCount_ == 0 || sums_.empty() || secondMoments_.empty()) {
        return Status::error(ErrorCode::NotInitialized,
                             "PaDiM pipeline contains no training samples");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to create PaDiM checkpoint", path.string());
    }
    stream.write(kCheckpointMagic.data(), static_cast<std::streamsize>(kCheckpointMagic.size()));
    const std::uint32_t dimension = static_cast<std::uint32_t>(config_.embeddingDimension);
    const std::uint32_t height = static_cast<std::uint32_t>(config_.featureHeight);
    const std::uint32_t width = static_cast<std::uint32_t>(config_.featureWidth);
    const std::uint64_t samples = static_cast<std::uint64_t>(sampleCount_);
    const std::uint64_t sumCount = static_cast<std::uint64_t>(sums_.size());
    const std::uint64_t momentCount = static_cast<std::uint64_t>(secondMoments_.size());
    if (!writeValue(stream, kCheckpointVersion) || !writeValue(stream, dimension) ||
        !writeValue(stream, height) || !writeValue(stream, width) ||
        !writeValue(stream, samples) || !writeValue(stream, sumCount) ||
        !writeValue(stream, momentCount)) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PaDiM checkpoint header", path.string());
    }
    stream.write(reinterpret_cast<const char*>(config_.channelIndices.data()),
                 static_cast<std::streamsize>(config_.channelIndices.size() * sizeof(std::int32_t)));
    stream.write(reinterpret_cast<const char*>(sums_.data()),
                 static_cast<std::streamsize>(sums_.size() * sizeof(double)));
    stream.write(reinterpret_cast<const char*>(secondMoments_.data()),
                 static_cast<std::streamsize>(secondMoments_.size() * sizeof(double)));
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PaDiM checkpoint", path.string());
    }
    return {};
}

Result<void> PadimPipeline::loadCheckpoint(
    const std::filesystem::path& path) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to open PaDiM checkpoint", path.string());
    }
    std::array<char, 8> magic{};
    std::uint32_t version = 0, dimension = 0, height = 0, width = 0;
    std::uint64_t samples = 0, sumCount = 0, momentCount = 0;
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || !readValue(stream, version) || !readValue(stream, dimension) ||
        !readValue(stream, height) || !readValue(stream, width) ||
        !readValue(stream, samples) || !readValue(stream, sumCount) ||
        !readValue(stream, momentCount) || magic != kCheckpointMagic ||
        version != kCheckpointVersion ||
        dimension != static_cast<std::uint32_t>(config_.embeddingDimension) ||
        height != static_cast<std::uint32_t>(config_.featureHeight) ||
        width != static_cast<std::uint32_t>(config_.featureWidth) || samples == 0 ||
        dimension == 0 || height == 0 || width == 0) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PaDiM checkpoint header is incompatible", path.string());
    }
    const std::size_t locations = static_cast<std::size_t>(height) * width;
    if (locations > std::numeric_limits<std::size_t>::max() / dimension) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PaDiM checkpoint dimensions overflow", path.string());
    }
    const std::size_t expectedSums = locations * dimension;
    if (expectedSums > std::numeric_limits<std::size_t>::max() / dimension) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PaDiM checkpoint dimensions overflow", path.string());
    }
    const std::size_t expectedMoments = expectedSums * dimension;
    if (sumCount != expectedSums || momentCount != expectedMoments ||
        samples > std::numeric_limits<std::size_t>::max()) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PaDiM checkpoint dimensions are invalid", path.string());
    }
    std::vector<std::int32_t> indices(dimension);
    std::vector<double> sums(expectedSums);
    std::vector<double> moments(expectedMoments);
    stream.read(reinterpret_cast<char*>(indices.data()),
                static_cast<std::streamsize>(indices.size() * sizeof(std::int32_t)));
    stream.read(reinterpret_cast<char*>(sums.data()),
                static_cast<std::streamsize>(sums.size() * sizeof(double)));
    stream.read(reinterpret_cast<char*>(moments.data()),
                static_cast<std::streamsize>(moments.size() * sizeof(double)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof() ||
        indices != config_.channelIndices) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PaDiM checkpoint payload is incompatible", path.string());
    }
    for (const double value : sums) {
        if (!std::isfinite(value)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PaDiM checkpoint contains non-finite sums", path.string());
        }
    }
    for (const double value : moments) {
        if (!std::isfinite(value)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PaDiM checkpoint contains non-finite moments", path.string());
        }
    }
    sampleCount_ = static_cast<std::size_t>(samples);
    sums_ = std::move(sums);
    secondMoments_ = std::move(moments);
    return {};
}

void PadimPipeline::clear() {
    sampleCount_ = 0;
    sums_.clear();
    secondMoments_.clear();
}

}  // namespace anom::model
