#pragma once

#include "backends/backend.h"

#include <memory>

namespace anom::model {

// ONNX Runtime backend. The implementation is intentionally hidden so public
// SDK headers do not expose ORT headers or ABI details to applications.
class OrtBackend final : public IRuntimeBackend {
public:
    OrtBackend();
    ~OrtBackend() override;
    OrtBackend(const OrtBackend&) = delete;
    OrtBackend& operator=(const OrtBackend&) = delete;
    OrtBackend(OrtBackend&&) noexcept;
    OrtBackend& operator=(OrtBackend&&) noexcept;

    Result<void> load(const BackendConfig& config) override;
    Result<TensorMap> infer(const TensorMap& inputs) override;
    [[nodiscard]] const TensorSignature& signature() const noexcept override;
    [[nodiscard]] int maxBatchSize() const noexcept override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
