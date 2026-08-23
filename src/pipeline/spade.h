#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <cstddef>
#include <filesystem>
#include <unordered_map>
#include <vector>

namespace anom::model {

struct SpadePipelineConfig {};

// Training-side SPADE pipeline. It accumulates patch features independently for
// every pyramid layer and writes one FAISS gallery per layer. The runtime
// adapter retrieves nearest neighbors from each gallery and fuses the resulting
// multi-resolution anomaly maps.
class SpadePipeline final {
public:
    explicit SpadePipeline(SpadePipelineConfig config = {});

    Result<void> addBackendOutputs(
        const TensorMap& outputs,
        const std::unordered_map<std::string, std::string>& outputBindings,
        const SPADEConfig& algorithmConfig);

    Result<void> saveIndexes(const std::vector<std::filesystem::path>& paths) const;

    void clear() noexcept;
    [[nodiscard]] std::size_t featureCount() const noexcept;

private:
    SpadePipelineConfig config_;
    std::vector<std::string> layerOrder_;
    std::unordered_map<std::string, std::vector<float>> layerFeatures_;
    std::unordered_map<std::string, int> layerChannels_;
};

}  // namespace anom::model
