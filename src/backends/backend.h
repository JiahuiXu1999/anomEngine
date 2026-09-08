#pragma once

#include "model/result.h"
#include "model/types.h"

#include <memory>

namespace anom::model {

struct BackendConfig {
    RuntimeBackend backend{RuntimeBackend::TensorRT};
    ExecutionProvider provider{ExecutionProvider::Default};
    std::filesystem::path onnxPath;
    std::filesystem::path enginePath;
    EngineLoadPolicy loadPolicy{EngineLoadPolicy::PreferEngine};
    bool fp16{false};
    int maxBatchSize{1};
    std::size_t workspaceBytes{2ULL * 1024ULL * 1024ULL * 1024ULL};
    std::string inputName{"input"};
    TensorLayout inputLayout{TensorLayout::NCHW};
    int inputHeight{224};
    int inputWidth{224};

    OrtGraphOptimization ortGraphOptimization{OrtGraphOptimization::All};
    OrtExecutionMode ortExecutionMode{OrtExecutionMode::Sequential};
    int deviceId{0};
    int intraOpThreads{0};
    int interOpThreads{0};
    bool enableMemoryPattern{true};
    bool enableCpuMemoryArena{true};
    bool enableProfiling{false};
    std::filesystem::path profileFilePrefix{"anom_ort_profile"};
};

class IRuntimeBackend {
public:
    virtual ~IRuntimeBackend() = default;
    virtual Result<void> load(const BackendConfig& config) = 0;
    // Probes runtime/device availability, not compatibility with a model.
    virtual Result<void> probe(ExecutionProvider provider, int deviceId) = 0;
    virtual Result<TensorMap> infer(const TensorMap& inputs) = 0;
    [[nodiscard]] virtual const TensorSignature& signature() const noexcept = 0;
    [[nodiscard]] virtual int maxBatchSize() const noexcept = 0;
};

}  // namespace anom::model
