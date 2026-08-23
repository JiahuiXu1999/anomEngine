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

template <typename T>
bool writeValue(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(T));
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

void PadimPipeline::clear() {
    sampleCount_ = 0;
    sums_.clear();
    secondMoments_.clear();
}

}  // namespace anom::model
