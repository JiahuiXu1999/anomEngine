#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <unordered_map>
#include <vector>

namespace anom::model {

class GpuCompute;

struct PadimPipelineConfig {
    int embeddingDimension{0};
    int featureHeight{0};
    int featureWidth{0};
    double covarianceRegularization{0.01};
    std::vector<std::int32_t> channelIndices;
    std::shared_ptr<GpuCompute> compute;
    // Limit the resident double statistics. Larger grids are processed in
    // location chunks while keeping the full selected embedding on the GPU.
    std::size_t gpuMemoryBudgetBytes{512u * 1024u * 1024u};
};

// Training-side PaDiM pipeline. It accumulates per-location Gaussian
// statistics from backend-neutral feature tensors and writes the artifacts
// consumed by PadimAdapter.
class PadimPipeline final {
public:
    explicit PadimPipeline(PadimPipelineConfig config);

    Result<void> addBackendOutputs(
        const TensorMap& outputs,
        const std::unordered_map<std::string, std::string>& outputBindings,
        const PadimConfig& algorithmConfig);
    Result<void> addEmbedding(const Tensor& fullEmbedding);
    Result<void> saveArtifacts(const std::filesystem::path& statisticsPath,
                               const std::filesystem::path& channelIndicesPath) const;
    Result<void> saveCheckpoint(const std::filesystem::path& path) const;
    Result<void> loadCheckpoint(const std::filesystem::path& path);

    void clear();
    [[nodiscard]] std::size_t sampleCount() const noexcept { return sampleCount_; }

private:
    Result<void> validateConfig() const;
    Result<void> addGpuEmbedding(const Tensor& selected);
    std::size_t gpuChunkLocations(std::size_t bytesPerLocation) const;

    PadimPipelineConfig config_;
    std::size_t sampleCount_{0};
    std::vector<double> sums_;
    std::vector<double> secondMoments_;
    Tensor deviceSums_;
    Tensor deviceSecondMoments_;
    // A failed CUDA operation can leave only some chunks updated. Require an
    // explicit clear or checkpoint restore rather than reuse partial state.
    bool gpuUpdateFailed_{false};
};

}  // namespace anom::model
