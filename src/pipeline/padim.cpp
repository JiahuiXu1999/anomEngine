#include "pipeline/padim.h"

#include "adapters/feature_utils.h"
#include "compute/gpu_compute.h"

#include <opencv2/core.hpp>

#include <algorithm>
#include <array>
#include <climits>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <set>
#include <stdexcept>

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

anom_compute_buffer hostDoubles(const double* data, std::size_t count) {
    return {const_cast<double*>(data), count * sizeof(double), -1};
}

anom_compute_buffer sliceBuffer(const Tensor& tensor, std::size_t offset,
                                std::size_t bytes) {
    auto buffer = GpuCompute::buffer(tensor);
    buffer.data = static_cast<std::byte*>(buffer.data) + offset;
    buffer.bytes = bytes;
    return buffer;
}

void zeroBuffer(const std::shared_ptr<GpuCompute>& compute, const Tensor& tensor) {
    const std::size_t chunkBytes = std::min<std::size_t>(tensor.byteSize(), 1024 * 1024);
    std::vector<std::byte> zeros(chunkBytes);
    for (std::size_t offset = 0; offset < tensor.byteSize(); offset += chunkBytes) {
        const auto bytes = std::min(chunkBytes, tensor.byteSize() - offset);
        compute->copyBuffer(sliceBuffer(tensor, offset, bytes), {zeros.data(), bytes, -1});
    }
}

Status gpuError(const char* operation, const std::exception& error) {
    const auto code = dynamic_cast<const std::bad_alloc*>(&error) ? ErrorCode::OutOfMemory
        : dynamic_cast<const std::invalid_argument*>(&error) ? ErrorCode::InvalidArgument
                                                            : ErrorCode::CudaFailure;
    return Status::error(code, operation, error.what());
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
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    const std::size_t locations = static_cast<std::size_t>(config_.featureHeight) *
                                  config_.featureWidth;
    if (locations > std::numeric_limits<std::size_t>::max() / dimension ||
        locations * dimension > std::numeric_limits<std::size_t>::max() / dimension /
                                    sizeof(double) ||
        locations * dimension * dimension >
            std::numeric_limits<std::size_t>::max() / sizeof(double) - locations * dimension) {
        return Status::error(ErrorCode::OutOfMemory,
                             "PaDiM statistics dimensions overflow size_t");
    }
    if (config_.compute && (locations > static_cast<std::size_t>(INT_MAX) ||
                            config_.gpuMemoryBudgetBytes == 0)) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PaDiM GPU locations or memory budget is invalid");
    }
    return {};
}

std::size_t PadimPipeline::gpuChunkLocations(std::size_t bytesPerLocation) const {
    const auto locations = static_cast<std::size_t>(config_.featureHeight) * config_.featureWidth;
    // A single location is the indivisible unit even when it exceeds the budget.
    return std::min(locations, std::max<std::size_t>(1, config_.gpuMemoryBudgetBytes / bytesPerLocation));
}

Result<void> PadimPipeline::addGpuEmbedding(const Tensor& selected) {
    if (gpuUpdateFailed_) {
        return Status::error(ErrorCode::NotInitialized,
                             "PaDiM GPU statistics require clear or checkpoint restore after a failed update");
    }
    const std::size_t locations = static_cast<std::size_t>(config_.featureHeight) * config_.featureWidth;
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    if (selected.shape.dims.size() != 4 || selected.shape.dims[0] <= 0 ||
        selected.shape.dims[0] > INT_MAX || selected.shape.dims[1] != config_.featureHeight ||
        selected.shape.dims[2] != config_.featureWidth ||
        selected.shape.dims[3] != config_.embeddingDimension) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "PaDiM GPU training embedding shape does not match configuration");
    }
    const auto batch = static_cast<std::size_t>(selected.shape.dims[0]);
    if (sampleCount_ > std::numeric_limits<std::size_t>::max() - batch) {
        return Status::error(ErrorCode::OutOfMemory, "PaDiM sample count overflows size_t");
    }
    try {
        const auto sumsCount = locations * dimension;
        const auto momentsCount = sumsCount * dimension;
        const auto bytesPerLocation = (dimension + dimension * dimension) * sizeof(double);
        if (!deviceSums_.deviceData && sums_.empty()) {
            if ((sumsCount + momentsCount) * sizeof(double) <= config_.gpuMemoryBudgetBytes) {
                auto sums = config_.compute->allocateBytes(sumsCount * sizeof(double));
                auto moments = config_.compute->allocateBytes(momentsCount * sizeof(double));
                zeroBuffer(config_.compute, sums);
                zeroBuffer(config_.compute, moments);
                deviceSums_ = std::move(sums);
                deviceSecondMoments_ = std::move(moments);
            } else {
                std::vector<double> sums(sumsCount, 0.0), moments(momentsCount, 0.0);
                sums_ = std::move(sums);
                secondMoments_ = std::move(moments);
            }
        }
        const auto chunk = gpuChunkLocations(bytesPerLocation);
        // selectFeatures validated every selected value before this point.
        // If any CUDA update fails, never silently continue with partial state.
        gpuUpdateFailed_ = true;
        for (std::size_t begin = 0; begin < locations; begin += chunk) {
            const auto count = std::min(chunk, locations - begin);
            const auto sums = deviceSums_.deviceData
                ? sliceBuffer(deviceSums_, begin * dimension * sizeof(double),
                              count * dimension * sizeof(double))
                : hostDoubles(sums_.data() + begin * dimension, count * dimension);
            const auto moments = deviceSecondMoments_.deviceData
                ? sliceBuffer(deviceSecondMoments_, begin * dimension * dimension * sizeof(double),
                              count * dimension * dimension * sizeof(double))
                : hostDoubles(secondMoments_.data() + begin * dimension * dimension,
                              count * dimension * dimension);
            config_.compute->padimAccumulate(selected, static_cast<int>(batch),
                static_cast<int>(locations), config_.embeddingDimension,
                static_cast<int>(begin), static_cast<int>(count), sums, moments);
        }
        sampleCount_ += batch;
        gpuUpdateFailed_ = false;
        return {};
    } catch (const std::exception& error) {
        return gpuError("Unable to accumulate PaDiM GPU statistics", error);
    }
}

Result<void> PadimPipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const PadimConfig& algorithmConfig) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    if (algorithmConfig.embeddingDimension != config_.embeddingDimension ||
        algorithmConfig.featureHeight != config_.featureHeight ||
        algorithmConfig.featureWidth != config_.featureWidth) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PaDiM pipeline and manifest dimensions differ");
    }
    if (config_.compute) {
        try {
            auto selected = config_.compute->selectFeatures(outputs, outputBindings,
                algorithmConfig.featureLayers, config_.channelIndices,
                config_.featureHeight, config_.featureWidth);
            return addGpuEmbedding(selected);
        } catch (const std::exception& error) {
            return gpuError("Unable to select PaDiM GPU training features", error);
        }
    }
    auto embedding = feature::aggregateFeaturePyramid(
        outputs, outputBindings, algorithmConfig.featureLayers, false, 1, 1, 0);
    if (!embedding) return embedding.status();
    return addEmbedding(embedding.value());
}

Result<void> PadimPipeline::addEmbedding(const Tensor& fullEmbedding) {
    auto valid = validateConfig();
    if (!valid) return valid.status();
    if (config_.compute) {
        try {
            // Preserve the public NCHW embedding contract while sharing the
            // fused selection path used for backend outputs.
            TensorMap outputs;
            auto& view = outputs["embedding"];
            view.dtype = fullEmbedding.dtype;
            view.shape = fullEmbedding.shape;
            if (fullEmbedding.deviceData) {
                view.deviceData = fullEmbedding.deviceData;
                view.deviceId = fullEmbedding.deviceId;
                view.externalByteSize = fullEmbedding.byteSize();
                view.externalOwner = fullEmbedding.externalOwner;
            } else {
                view.setExternalView(fullEmbedding.data<std::byte>(), fullEmbedding.byteSize());
            }
            auto selected = config_.compute->selectFeatures(outputs, {{"embedding", "embedding"}},
                {"embedding"}, config_.channelIndices, config_.featureHeight, config_.featureWidth);
            return addGpuEmbedding(selected);
        } catch (const std::exception& error) {
            return gpuError("Unable to select PaDiM GPU training embedding", error);
        }
    }
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
    for (const float value : patches) {
        if (!std::isfinite(value)) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "PaDiM training embedding contains non-finite values");
        }
    }
    for (int batch = 0; batch < view.value().batch; ++batch) {
        for (std::size_t location = 0; location < locations; ++location) {
            const float* patch = patches.data() +
                (static_cast<std::size_t>(batch) * locations + location) * dimension;
            double* sum = sums_.data() + location * dimension;
            double* moment = secondMoments_.data() + location * dimension * dimension;
            for (std::size_t row = 0; row < dimension; ++row) {
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
    if (gpuUpdateFailed_) {
        return Status::error(ErrorCode::NotInitialized,
                             "PaDiM GPU statistics require clear or checkpoint restore after a failed update");
    }
    if (sampleCount_ == 0 ||
        (!deviceSums_.deviceData && (sums_.empty() || secondMoments_.empty()))) {
        return Status::error(ErrorCode::NotInitialized,
                             "PaDiM pipeline contains no training samples");
    }

    const std::size_t locations = static_cast<std::size_t>(config_.featureHeight) *
                                  config_.featureWidth;
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    std::vector<float> means(locations * dimension);
    std::vector<float> precisions(locations * dimension * dimension);
    if (config_.compute) {
        try {
            // Finalization needs covariance/inverse workspace in addition to
            // the input statistics. Bound each solve batch independently of
            // whether the accumulated statistics are resident or host backed.
            const auto bytesPerLocation = (dimension + dimension * dimension) * sizeof(double);
            const auto chunk = std::min(locations, std::max<std::size_t>(1,
                config_.gpuMemoryBudgetBytes / 4 / bytesPerLocation));
            for (std::size_t begin = 0; begin < locations; begin += chunk) {
                const auto count = std::min(chunk, locations - begin);
                const auto sums = deviceSums_.deviceData
                    ? sliceBuffer(deviceSums_, begin * dimension * sizeof(double),
                                  count * dimension * sizeof(double))
                    : hostDoubles(sums_.data() + begin * dimension, count * dimension);
                const auto moments = deviceSecondMoments_.deviceData
                    ? sliceBuffer(deviceSecondMoments_, begin * dimension * dimension * sizeof(double),
                                  count * dimension * dimension * sizeof(double))
                    : hostDoubles(secondMoments_.data() + begin * dimension * dimension,
                                  count * dimension * dimension);
                config_.compute->padimFinalize(sums, moments, static_cast<int>(count),
                    config_.embeddingDimension, static_cast<std::uint64_t>(sampleCount_),
                    config_.covarianceRegularization, means.data() + begin * dimension,
                    precisions.data() + begin * dimension * dimension);
            }
        } catch (const std::exception& error) {
            return gpuError("Unable to finalize PaDiM GPU statistics", error);
        }
    } else {
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
    if (gpuUpdateFailed_) {
        return Status::error(ErrorCode::NotInitialized,
                             "PaDiM GPU statistics require clear or checkpoint restore after a failed update");
    }
    if (sampleCount_ == 0 ||
        (!deviceSums_.deviceData && (sums_.empty() || secondMoments_.empty()))) {
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
    const std::uint64_t sumCount = static_cast<std::uint64_t>(height) * width * dimension;
    const std::uint64_t momentCount = sumCount * dimension;
    if (!writeValue(stream, kCheckpointVersion) || !writeValue(stream, dimension) ||
        !writeValue(stream, height) || !writeValue(stream, width) ||
        !writeValue(stream, samples) || !writeValue(stream, sumCount) ||
        !writeValue(stream, momentCount)) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PaDiM checkpoint header", path.string());
    }
    stream.write(reinterpret_cast<const char*>(config_.channelIndices.data()),
                 static_cast<std::streamsize>(config_.channelIndices.size() * sizeof(std::int32_t)));
    if (deviceSums_.deviceData) {
        try {
            // Preserve the v1 checkpoint byte layout without allocating a
            // second full host copy of resident double statistics.
            const std::size_t chunkBytes = std::min<std::size_t>(config_.gpuMemoryBudgetBytes, 8u * 1024u * 1024u);
            std::vector<std::byte> chunk(chunkBytes);
            for (const auto* tensor : {&deviceSums_, &deviceSecondMoments_}) {
                for (std::size_t offset = 0; offset < tensor->byteSize(); offset += chunkBytes) {
                    const auto bytes = std::min(chunkBytes, tensor->byteSize() - offset);
                    config_.compute->copyBuffer({chunk.data(), bytes, -1}, sliceBuffer(*tensor, offset, bytes));
                    stream.write(reinterpret_cast<const char*>(chunk.data()), static_cast<std::streamsize>(bytes));
                }
            }
        } catch (const std::exception& error) {
            return gpuError("Unable to snapshot PaDiM GPU statistics", error);
        }
    } else {
        stream.write(reinterpret_cast<const char*>(sums_.data()),
                     static_cast<std::streamsize>(sums_.size() * sizeof(double)));
        stream.write(reinterpret_cast<const char*>(secondMoments_.data()),
                     static_cast<std::streamsize>(secondMoments_.size() * sizeof(double)));
    }
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
    Tensor deviceSums, deviceMoments;
    if (config_.compute && (sums.size() + moments.size()) * sizeof(double) <= config_.gpuMemoryBudgetBytes) {
        try {
            deviceSums = config_.compute->allocateBytes(sums.size() * sizeof(double));
            deviceMoments = config_.compute->allocateBytes(moments.size() * sizeof(double));
            config_.compute->copyBuffer(GpuCompute::buffer(deviceSums), hostDoubles(sums.data(), sums.size()));
            config_.compute->copyBuffer(GpuCompute::buffer(deviceMoments), hostDoubles(moments.data(), moments.size()));
        } catch (const std::exception& error) {
            return gpuError("Unable to restore PaDiM GPU statistics", error);
        }
        std::vector<double>().swap(sums);
        std::vector<double>().swap(moments);
    }
    sampleCount_ = static_cast<std::size_t>(samples);
    sums_ = std::move(sums);
    secondMoments_ = std::move(moments);
    deviceSums_ = std::move(deviceSums);
    deviceSecondMoments_ = std::move(deviceMoments);
    gpuUpdateFailed_ = false;
    return {};
}

void PadimPipeline::clear() {
    sampleCount_ = 0;
    sums_.clear();
    secondMoments_.clear();
    deviceSums_ = {};
    deviceSecondMoments_ = {};
    gpuUpdateFailed_ = false;
}

}  // namespace anom::model
