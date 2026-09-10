#pragma once

#include "adapters/adapter.h"

#include <memory>

namespace anom::model {

// Inference-side SPADE adapter. It retrieves nearest neighbors from a per-layer
// FAISS gallery, builds a per-layer anomaly map, and fuses the multi-resolution
// maps into a single anomaly map.
class SpadeAdapter final : public IModelAdapter {
public:
    SpadeAdapter(SPADEConfig config,
                 std::unordered_map<std::string, std::string> outputBindings);
    ~SpadeAdapter() override;
    SpadeAdapter(const SpadeAdapter&) = delete;
    SpadeAdapter& operator=(const SpadeAdapter&) = delete;

    Result<void> loadAssets(const ModelPackage& package) override;
    Result<SearchExecutionInfo> configureSearch(const SearchExecutionConfig& config) override;
    Result<void> validateSignature(const TensorSignature& signature) const override;
    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
