#include "backends/backend_factory.h"

#include "backends/tensorrt_backend.h"

#if defined(ANOM_ENABLE_ONNXRUNTIME)
#include "backends/ort_backend.h"
#endif

namespace anom::model {

Result<std::unique_ptr<IRuntimeBackend>> createRuntimeBackend(RuntimeBackend backend) {
    switch (backend) {
        case RuntimeBackend::TensorRT:
            return std::unique_ptr<IRuntimeBackend>(new TensorRTBackend());
        case RuntimeBackend::OnnxRuntime:
#if defined(ANOM_ENABLE_ONNXRUNTIME)
            return std::unique_ptr<IRuntimeBackend>(new OrtBackend());
#else
            return Status::error(
                ErrorCode::BackendFailure,
                "Model requests ONNX Runtime but anom_model was built without it",
                "Configure with -DANOM_ENABLE_ONNXRUNTIME=ON and set ONNXRUNTIME_ROOT");
#endif
    }
    return Status::error(ErrorCode::InvalidArgument, "Unknown runtime backend");
}

}  // namespace anom::model
