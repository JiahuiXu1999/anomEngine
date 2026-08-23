#pragma once

#include "adapters/adapter.h"

namespace anom::model {

class DirectPredictionAdapter final : public IModelAdapter {
public:
    DirectPredictionAdapter(DirectConfig config,
                            std::unordered_map<std::string, std::string> outputBindings)
        : config_(std::move(config)), outputBindings_(std::move(outputBindings)) {}

    Result<void> loadAssets(const ModelPackage&) override { return {}; }
    Result<void> validateSignature(const TensorSignature& signature) const override;
    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override;

private:
    DirectConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
};

}  // namespace anom::model
