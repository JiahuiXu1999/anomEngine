#pragma once

#include "adapters/adapter.h"

#include <memory>

namespace anom::model {

class PatchCoreAdapter final : public IModelAdapter {
public:
    PatchCoreAdapter(PatchCoreConfig config,
                     std::unordered_map<std::string, std::string> outputBindings);
    ~PatchCoreAdapter() override;
    PatchCoreAdapter(const PatchCoreAdapter&) = delete;
    PatchCoreAdapter& operator=(const PatchCoreAdapter&) = delete;

    Result<void> loadAssets(const ModelPackage& package) override;
    Result<void> validateSignature(const TensorSignature& signature) const override;
    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
