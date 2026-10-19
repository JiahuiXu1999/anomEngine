#pragma once

#include "adapters/adapter.h"

#include <memory>

namespace anom::model {

// Inference-side DFKDE adapter. It projects aggregated pyramid features with
// the stored PCA basis and scores each patch by the negative log kernel density
// against the stored normal gallery.
class DfkdeAdapter final : public IModelAdapter {
public:
    DfkdeAdapter(DFKDEConfig config,
                 std::unordered_map<std::string, std::string> outputBindings);
    ~DfkdeAdapter() override;
    DfkdeAdapter(const DfkdeAdapter&) = delete;
    DfkdeAdapter& operator=(const DfkdeAdapter&) = delete;

    Result<void> loadAssets(const ModelPackage& package) override;
    Result<void> configureCompute(int device, const std::filesystem::path& directory) override;
    bool supportsDeviceCompute() const noexcept override { return true; }
    Result<void> validateSignature(const TensorSignature& signature) const override;
    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
