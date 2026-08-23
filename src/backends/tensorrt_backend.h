#pragma once

#include "backends/backend.h"

#include <memory>

namespace anom::model {

class TensorRTBackend final : public IRuntimeBackend {
public:
    TensorRTBackend();
    ~TensorRTBackend() override;
    TensorRTBackend(const TensorRTBackend&) = delete;
    TensorRTBackend& operator=(const TensorRTBackend&) = delete;
    TensorRTBackend(TensorRTBackend&&) noexcept;
    TensorRTBackend& operator=(TensorRTBackend&&) noexcept;

    Result<void> load(const BackendConfig& config) override;
    Result<TensorMap> infer(const TensorMap& inputs) override;
    [[nodiscard]] const TensorSignature& signature() const noexcept override;
    [[nodiscard]] int maxBatchSize() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
