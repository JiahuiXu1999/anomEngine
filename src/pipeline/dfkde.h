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

struct DfkdePipelineConfig {
    int embeddingDimension{0};
    int nComponents{16};
    float kernelSigma{10.0F};
};

// Training-side DFKDE pipeline. It accumulates normal patch embeddings, reduces
// them with PCA, and stores the projected gallery plus the projection basis as
// a single statistics artifact consumed by DfkdeAdapter.
class DfkdePipeline final {
public:
    explicit DfkdePipeline(DfkdePipelineConfig config);

    Result<void> addBackendOutputs(
        const TensorMap& outputs,
        const std::unordered_map<std::string, std::string>& outputBindings,
        const DFKDEConfig& algorithmConfig);
    Result<void> addEmbedding(const Tensor& embedding);
    Result<void> addFeatures(const std::vector<float>& features,
                             std::size_t vectorCount);

    Result<void> saveStatistics(const std::filesystem::path& path) const;

    void clear() noexcept;
    [[nodiscard]] std::size_t featureCount() const noexcept;

private:
    Result<void> validateConfig() const;

    DfkdePipelineConfig config_;
    std::vector<float> features_;
};

}  // namespace anom::model
