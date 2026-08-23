#include "pipeline/patchcore.h"

#include "adapters/feature_utils.h"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>

namespace anom::model {
namespace {

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
    : config_(std::move(config)) {}

Result<void> PatchCorePipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const PatchCoreConfig& algorithmConfig) {
    if (algorithmConfig.embeddingDimension != config_.embeddingDimension) {
        return Status::error(ErrorCode::InvalidArgument,
                             "PatchCore pipeline and manifest embedding dimensions differ");
    }
    auto embedding = feature::aggregateFeaturePyramid(
        outputs, outputBindings, algorithmConfig.featureLayers, true,
        algorithmConfig.poolingKernel, algorithmConfig.poolingStride,
        algorithmConfig.poolingPadding);
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

void PatchCorePipeline::clear() noexcept { features_.clear(); }

std::size_t PatchCorePipeline::featureCount() const noexcept {
    return config_.embeddingDimension > 0
               ? features_.size() / static_cast<std::size_t>(config_.embeddingDimension)
               : 0;
}

int PatchCorePipeline::embeddingDimension() const noexcept {
    return config_.embeddingDimension;
}

}  // namespace anom::model
