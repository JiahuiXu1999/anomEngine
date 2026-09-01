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

struct PadimPipelineConfig {
    int embeddingDimension{0};
    int featureHeight{0};
    int featureWidth{0};
    double covarianceRegularization{0.01};
    std::vector<std::int32_t> channelIndices;
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

    PadimPipelineConfig config_;
    std::size_t sampleCount_{0};
    std::vector<double> sums_;
    std::vector<double> secondMoments_;
};

}  // namespace anom::model
