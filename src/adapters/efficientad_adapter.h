#pragma once

#include "adapters/adapter.h"

#include <memory>

namespace anom::model {

// Inference-side EfficientAD adapter. The backend returns a teacher feature
// stream, a student feature stream, and a reconstruction-error map. The adapter
// derives the hard score from the teacher/student discrepancy and combines it
// with the soft reconstruction error into a single anomaly map.
class EfficientADAdapter final : public IModelAdapter {
public:
    EfficientADAdapter(EfficientADConfig config,
                       std::unordered_map<std::string, std::string> outputBindings);
    ~EfficientADAdapter() override;
    EfficientADAdapter(const EfficientADAdapter&) = delete;
    EfficientADAdapter& operator=(const EfficientADAdapter&) = delete;

    Result<void> loadAssets(const ModelPackage& package) override;
    Result<void> validateSignature(const TensorSignature& signature) const override;
    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
