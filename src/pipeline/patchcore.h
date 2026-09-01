#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace anom::model {

struct PatchCorePipelineConfig {
    int embeddingDimension{0};
    float coresetSamplingRatio{0.1F};
    std::size_t projectionDimension{128};
    std::size_t maxExactDistanceEvaluations{50'000'000};
    std::uint32_t randomSeed{42};
};

// Training-side PatchCore algorithm pipeline. Runtime inference remains in
// PatchCoreAdapter; this class consumes backend-neutral feature tensors and
// creates the immutable FAISS memory-bank artifact used by the adapter.
class PatchCorePipeline final {
public:
    explicit PatchCorePipeline(PatchCorePipelineConfig config);

    Result<void> addBackendOutputs(
        const TensorMap& outputs,
        const std::unordered_map<std::string, std::string>& outputBindings,
        const PatchCoreConfig& algorithmConfig);
    Result<void> addEmbedding(const Tensor& embedding);
    Result<void> addFeatures(const std::vector<float>& features,
                             std::size_t vectorCount);

    Result<std::vector<float>> buildCoreset() const;
    Result<std::size_t> loadMemoryBank(const std::filesystem::path& path);
    Result<std::size_t> saveMemoryBank(const std::filesystem::path& path) const;
    Result<void> saveCheckpoint(const std::filesystem::path& path) const;
    Result<void> loadCheckpoint(const std::filesystem::path& path);

    void clear() noexcept;
    [[nodiscard]] std::size_t featureCount() const noexcept;
    [[nodiscard]] int embeddingDimension() const noexcept;

private:
    PatchCorePipelineConfig config_;
    std::vector<float> features_;
};

}  // namespace anom::model
