#include "pipeline/patchcore.h"
#include "compute/gpu_compute.h"
#include "pipeline/patchcore_statistics.h"

#include "adapters/feature_utils.h"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <memory>
#include <numeric>
#include <random>
#include <stdexcept>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kCheckpointMagic{'A', 'N', 'P', 'C', 'K', 'P', 'T', '\0'};
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

Result<void> validateConfig(const PatchCorePipelineConfig& config) {
    if (config.embeddingDimension <= 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore embedding dimension must be positive");
    }
    if (!std::isfinite(config.coresetSamplingRatio) ||
        config.coresetSamplingRatio <= 0.0F || config.coresetSamplingRatio > 1.0F) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore coreset sampling ratio must be in (0, 1]");
    }
    if (config.projectionDimension == 0 || config.maxExactDistanceEvaluations == 0) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore projection and distance limits must be positive");
    }
    return {};
}

std::vector<float> projectFeatures(const std::vector<float>& features,
                                   std::size_t vectorCount,
                                   std::size_t inputDimension,
                                   std::size_t outputDimension,
                                   std::uint32_t seed) {
    if (outputDimension >= inputDimension) return features;

    std::mt19937 random(seed);
    std::normal_distribution<float> normal(
        0.0F, 1.0F / std::sqrt(static_cast<float>(outputDimension)));
    std::vector<float> projection(inputDimension * outputDimension);
    for (auto& value : projection) value = normal(random);

    std::vector<float> projected(vectorCount * outputDimension, 0.0F);
    for (std::size_t vector = 0; vector < vectorCount; ++vector) {
        const float* source = features.data() + vector * inputDimension;
        float* target = projected.data() + vector * outputDimension;
        for (std::size_t output = 0; output < outputDimension; ++output) {
            double sum = 0.0;
            for (std::size_t input = 0; input < inputDimension; ++input) {
                sum += static_cast<double>(source[input]) *
                       projection[input * outputDimension + output];
            }
            target[output] = static_cast<float>(sum);
        }
    }
    return projected;
}

}  // namespace

PatchCorePipeline::PatchCorePipeline(PatchCorePipelineConfig config)
    : config_(std::move(config)) {
    if (config_.statisticalConfig.variant != PatchCoreVariant::Original) {
        config_.statisticalConfig.embeddingDimension = config_.embeddingDimension;
        statistical_ = std::make_unique<PatchCoreStatisticalFit>(config_.statisticalConfig, config_.compute);
    }
}
PatchCorePipeline::~PatchCorePipeline() = default;

Result<void> PatchCorePipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const PatchCoreConfig& algorithmConfig) {
    if (algorithmConfig.embeddingDimension != config_.embeddingDimension) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore pipeline and manifest embedding dimensions differ");
    }
    if (config_.compute) {
        try {
            const auto device = config_.compute->aggregate(outputs, outputBindings, algorithmConfig);
            const auto host = config_.compute->download(device);
            const auto* data = host.data<float>();
            return addFeatures(std::vector<float>(data, data + host.byteSize() / sizeof(float)),
                               host.byteSize() / sizeof(float) / config_.embeddingDimension);
        } catch (const std::exception& e) {
            return Status::error(ErrorCode::CudaFailure, "GPU training aggregation failed", e.what());
        }
    }
    auto embedding = aggregatePatchCoreFeatures(outputs, outputBindings, algorithmConfig);
    if (!embedding) return embedding.status();
    return addEmbedding(embedding.value());
}

Result<void> PatchCorePipeline::addEmbedding(const Tensor& embedding) {
    auto valid = validateConfig(config_);
    if (!valid) return valid.status();
    auto view = feature::viewNchwFloat(embedding, "PatchCore training embedding");
    if (!view) return view.status();
    if (view.value().channels != config_.embeddingDimension) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "PatchCore training embedding has the wrong channel count",
                             embedding.shape.toString());
    }
    const auto flattened = feature::flattenPatches(embedding);
    const std::size_t count = static_cast<std::size_t>(view.value().batch) *
                              view.value().height * view.value().width;
    return addFeatures(flattened, count);
}

Result<void> PatchCorePipeline::addFeatures(const std::vector<float>& features,
                                            std::size_t vectorCount) {
    if (statistical_) return statistical_->add(features, vectorCount);
    auto valid = validateConfig(config_);
    if (!valid) return valid.status();
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    if (vectorCount == 0 || vectorCount > std::numeric_limits<std::size_t>::max() / dimension ||
        features.size() != vectorCount * dimension) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore feature matrix has an invalid size");
    }
    for (const float value : features) {
        if (!std::isfinite(value)) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "PatchCore feature matrix contains non-finite values");
        }
    }
    if (features_.size() > std::numeric_limits<std::size_t>::max() - features.size()) {
        return Status::error(ErrorCode::OutOfMemory,
                             "PatchCore feature matrix size overflows size_t");
    }
    features_.insert(features_.end(), features.begin(), features.end());
    return {};
}

Result<std::vector<float>> PatchCorePipeline::buildCoreset() const {
    if (statistical_) return Status::error(ErrorCode::UnsupportedAlgorithm,
        "Statistical PatchCore variants require saveArtifacts");
    auto valid = validateConfig(config_);
    if (!valid) return valid.status();
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    const std::size_t vectorCount = features_.size() / dimension;
    if (vectorCount == 0) {
        return Status::error(ErrorCode::NotInitialized,
                             "PatchCore pipeline contains no training features");
    }
    const std::size_t targetCount = std::max<std::size_t>(
        1, static_cast<std::size_t>(std::ceil(
               static_cast<double>(vectorCount) * config_.coresetSamplingRatio)));
    if (targetCount >= vectorCount) return features_;

    if (config_.compute) {
        try {
            if (vectorCount > INT_MAX || dimension > INT_MAX || targetCount > INT_MAX)
                return Status::error(ErrorCode::InvalidArgument, "GPU coreset exceeds int32 dimensions");
            const auto rank = std::min(config_.projectionDimension, dimension);
            Tensor selection;
            if (rank < dimension) {
                std::mt19937 random(config_.randomSeed);
                std::normal_distribution<float> normal(0.0F, 1.0F / std::sqrt(static_cast<float>(rank)));
                std::vector<float> projection(dimension * rank);
                for (auto& value : projection) value = normal(random);
                selection = config_.compute->project(features_, static_cast<int>(vectorCount),
                    static_cast<int>(dimension), projection, static_cast<int>(rank));
            } else {
                Tensor input;
                input.shape.dims = {static_cast<int64_t>(vectorCount), static_cast<int64_t>(dimension)};
                input.setExternalView(features_.data(), features_.size() * sizeof(float));
                selection = config_.compute->upload(input);
            }
            std::mt19937 random(config_.randomSeed);
            std::uniform_int_distribution<std::size_t> first(0, vectorCount - 1);
            auto indices = config_.compute->coreset(selection, static_cast<int>(targetCount), static_cast<int>(first(random)));
            std::vector<float> coreset;
            coreset.reserve(targetCount * dimension);
            for (auto index : indices) {
                const auto* begin = features_.data() + static_cast<std::size_t>(index) * dimension;
                coreset.insert(coreset.end(), begin, begin + dimension);
            }
            return coreset;
        } catch (const std::exception& e) {
            return Status::error(ErrorCode::CudaFailure, "GPU coreset failed", e.what());
        }
    }

    if (targetCount > config_.maxExactDistanceEvaluations / vectorCount) {
        std::vector<std::size_t> candidates(vectorCount);
        std::iota(candidates.begin(), candidates.end(), std::size_t{0});
        std::mt19937 random(config_.randomSeed);
        std::shuffle(candidates.begin(), candidates.end(), random);
        std::vector<float> coreset;
        coreset.reserve(targetCount * dimension);
        for (std::size_t selected = 0; selected < targetCount; ++selected) {
            const float* source = features_.data() + candidates[selected] * dimension;
            coreset.insert(coreset.end(), source, source + dimension);
        }
        return coreset;
    }

    const std::size_t projectionDimension =
        std::min(config_.projectionDimension, dimension);
    const std::vector<float> selectionFeatures = projectFeatures(
        features_, vectorCount, dimension, projectionDimension, config_.randomSeed);
    std::mt19937 random(config_.randomSeed);
    std::uniform_int_distribution<std::size_t> firstDistribution(0, vectorCount - 1);
    const std::size_t first = firstDistribution(random);
    std::vector<std::size_t> selected;
    selected.reserve(targetCount);
    selected.push_back(first);
    std::vector<bool> isSelected(vectorCount, false);
    isSelected[first] = true;
    std::vector<double> minimumDistances(vectorCount,
                                         std::numeric_limits<double>::infinity());

    const auto squaredDistance = [&](std::size_t left, std::size_t right) {
        const float* a = selectionFeatures.data() + left * projectionDimension;
        const float* b = selectionFeatures.data() + right * projectionDimension;
        double sum = 0.0;
        for (std::size_t index = 0; index < projectionDimension; ++index) {
            const double difference = static_cast<double>(a[index]) - b[index];
            sum += difference * difference;
        }
        return sum;
    };

    for (std::size_t index = 0; index < vectorCount; ++index) {
        minimumDistances[index] = squaredDistance(index, first);
    }
    minimumDistances[first] = 0.0;
    while (selected.size() < targetCount) {
        std::size_t best = vectorCount;
        double bestDistance = -1.0;
        for (std::size_t index = 0; index < vectorCount; ++index) {
            if (!isSelected[index] && minimumDistances[index] > bestDistance) {
                best = index;
                bestDistance = minimumDistances[index];
            }
        }
        if (best == vectorCount) break;
        selected.push_back(best);
        isSelected[best] = true;
        for (std::size_t index = 0; index < vectorCount; ++index) {
            if (!isSelected[index]) {
                minimumDistances[index] = std::min(
                    minimumDistances[index], squaredDistance(index, best));
            }
        }
    }

    std::vector<float> coreset;
    coreset.reserve(selected.size() * dimension);
    for (const auto index : selected) {
        const float* source = features_.data() + index * dimension;
        coreset.insert(coreset.end(), source, source + dimension);
    }
    return coreset;
}

Result<std::size_t> PatchCorePipeline::loadMemoryBank(
    const std::filesystem::path& path) {
    if (statistical_) return Status::error(ErrorCode::UnsupportedAlgorithm,
        "Restore statistical fitting from a checkpoint, not a transformed memory bank");
    auto valid = validateConfig(config_);
    if (!valid) return valid.status();
    try {
        std::unique_ptr<faiss::Index> index(faiss::read_index(path.string().c_str()));
        if (!index || index->d != config_.embeddingDimension || index->ntotal <= 0) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PatchCore memory bank is incompatible", path.string());
        }
        const std::size_t count = static_cast<std::size_t>(index->ntotal);
        if (count > std::numeric_limits<std::size_t>::max() /
                        static_cast<std::size_t>(config_.embeddingDimension)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PatchCore memory bank dimensions overflow", path.string());
        }
        std::vector<float> loaded(count * static_cast<std::size_t>(config_.embeddingDimension));
        index->reconstruct_n(0, index->ntotal, loaded.data());
        auto added = addFeatures(loaded, count);
        if (!added) return added.status();
        return count;
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "Unable to load PatchCore memory bank", error.what());
    }
}

Result<std::size_t> PatchCorePipeline::saveMemoryBank(
    const std::filesystem::path& path) const {
    auto coreset = buildCoreset();
    if (!coreset) return coreset.status();
    const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
    const std::size_t count = coreset.value().size() / dimension;
    if (count > static_cast<std::size_t>(std::numeric_limits<faiss::idx_t>::max())) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore coreset exceeds the FAISS index range");
    }
    try {
        faiss::IndexFlatL2 index(config_.embeddingDimension);
        index.add(static_cast<faiss::idx_t>(count), coreset.value().data());
        faiss::write_index(&index, path.string().c_str());
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::IoError,
                             "Unable to save PatchCore memory bank", error.what());
    }
    return count;
}

Result<void> PatchCorePipeline::saveCheckpoint(
    const std::filesystem::path& path) const {
    if (statistical_) return statistical_->saveCheckpoint(path);
    auto valid = validateConfig(config_);
    if (!valid) return valid.status();
    if (features_.empty()) {
        return Status::error(ErrorCode::NotInitialized,
                             "PatchCore pipeline contains no training features");
    }
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to create PatchCore checkpoint", path.string());
    }
    stream.write(kCheckpointMagic.data(), static_cast<std::streamsize>(kCheckpointMagic.size()));
    const std::int32_t dimension = config_.embeddingDimension;
    const std::uint64_t valueCount = static_cast<std::uint64_t>(features_.size());
    if (!writeValue(stream, kCheckpointVersion) || !writeValue(stream, dimension) ||
        !writeValue(stream, valueCount)) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PatchCore checkpoint header", path.string());
    }
    stream.write(reinterpret_cast<const char*>(features_.data()),
                 static_cast<std::streamsize>(features_.size() * sizeof(float)));
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write PatchCore checkpoint", path.string());
    }
    return {};
}

Result<void> PatchCorePipeline::loadCheckpoint(
    const std::filesystem::path& path) {
    if (statistical_) return statistical_->loadCheckpoint(path);
    auto valid = validateConfig(config_);
    if (!valid) return valid.status();
    std::ifstream stream(path, std::ios::binary);
    if (!stream) {
        return Status::error(ErrorCode::IoError,
                             "Unable to open PatchCore checkpoint", path.string());
    }
    std::array<char, 8> magic{};
    std::uint32_t version = 0;
    std::int32_t dimension = 0;
    std::uint64_t valueCount = 0;
    stream.read(magic.data(), static_cast<std::streamsize>(magic.size()));
    if (!stream || !readValue(stream, version) || !readValue(stream, dimension) ||
        !readValue(stream, valueCount) || magic != kCheckpointMagic ||
        version != kCheckpointVersion || dimension != config_.embeddingDimension ||
        valueCount == 0 || valueCount % static_cast<std::uint64_t>(dimension) != 0 ||
        valueCount > std::numeric_limits<std::size_t>::max()) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PatchCore checkpoint header is incompatible", path.string());
    }
    std::vector<float> loaded(static_cast<std::size_t>(valueCount));
    stream.read(reinterpret_cast<char*>(loaded.data()),
                static_cast<std::streamsize>(loaded.size() * sizeof(float)));
    if (!stream || stream.peek() != std::ifstream::traits_type::eof()) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PatchCore checkpoint payload is invalid", path.string());
    }
    for (const float value : loaded) {
        if (!std::isfinite(value)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PatchCore checkpoint contains non-finite values", path.string());
        }
    }
    features_ = std::move(loaded);
    return {};
}

Result<std::size_t> PatchCorePipeline::saveArtifacts(
    const std::filesystem::path& index, const std::filesystem::path& statistics) const {
    if (statistical_) return statistical_->saveArtifacts(index, statistics);
    return saveMemoryBank(index);
}

void PatchCorePipeline::clear() noexcept {
    features_.clear();
    if (statistical_) statistical_->clear();
}

std::size_t PatchCorePipeline::featureCount() const noexcept {
    if (statistical_) return statistical_->count();
    return config_.embeddingDimension > 0
               ? features_.size() / static_cast<std::size_t>(config_.embeddingDimension)
               : 0;
}

int PatchCorePipeline::embeddingDimension() const noexcept {
    return config_.embeddingDimension;
}

}  // namespace anom::model
