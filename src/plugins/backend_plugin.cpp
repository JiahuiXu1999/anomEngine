#include "plugin_api/anom_backend_plugin.h"

#include "backends/backend.h"
#include "infrastructure/utf8_path.h"

#if defined(ANOM_BACKEND_TENSORRT)
#  include "backends/tensorrt_backend.h"
#elif defined(ANOM_BACKEND_ONNXRUNTIME)
#  include "backends/ort_backend.h"
#else
#  error "A backend plugin target must define its backend"
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <memory>
#include <new>
#include <string>
#include <vector>

namespace {
using namespace anom::model;

thread_local std::string globalError;

struct Instance {
    std::unique_ptr<IRuntimeBackend> backend;
    std::vector<anom_plugin_tensor_spec_v1> inputViews;
    std::vector<anom_plugin_tensor_spec_v1> outputViews;
    std::string lastError;
};

struct BatchOwner {
    TensorMap tensors;
    std::vector<anom_plugin_tensor_view_v1> views;
};

int32_t fail(Instance* instance, const Status& status) {
    if (instance) instance->lastError = status.describe();
    else globalError = status.describe();
    return static_cast<int32_t>(status.code);
}

bool validDataType(int32_t type) {
    return type >= static_cast<int32_t>(DataType::Float32) &&
           type <= static_cast<int32_t>(DataType::Bool);
}

std::unique_ptr<IRuntimeBackend> makeBackend() {
#if defined(ANOM_BACKEND_TENSORRT)
    return std::make_unique<TensorRTBackend>();
#elif defined(ANOM_BACKEND_ONNXRUNTIME)
    return std::make_unique<OrtBackend>();
#endif
}

int32_t ANOM_PLUGIN_CALL create(const anom_backend_config_v1* source, void** outInstance) {
    globalError.clear();
    if (!source || source->struct_size < sizeof(anom_backend_config_v1) || !outInstance) {
        return fail(nullptr, Status::error(ErrorCode::InvalidArgument,
                                           "Backend plugin create arguments are invalid"));
    }
    *outInstance = nullptr;
    try {
        BackendConfig config;
        if (source->onnx_path_utf8 && *source->onnx_path_utf8)
            config.onnxPath = pathFromUtf8(source->onnx_path_utf8);
        if (source->engine_path_utf8 && *source->engine_path_utf8)
            config.enginePath = pathFromUtf8(source->engine_path_utf8);
        config.loadPolicy = static_cast<EngineLoadPolicy>(source->load_policy);
        config.fp16 = source->fp16 != 0;
        config.maxBatchSize = source->max_batch_size;
        config.workspaceBytes = static_cast<std::size_t>(source->workspace_bytes);
        config.inputName = source->input_name_utf8 ? source->input_name_utf8 : "input";
        config.inputLayout = static_cast<TensorLayout>(source->input_layout);
        config.inputHeight = source->input_height;
        config.inputWidth = source->input_width;
        config.ortProvider = static_cast<OrtExecutionProvider>(source->ort_provider);
        config.ortGraphOptimization =
            static_cast<OrtGraphOptimization>(source->ort_graph_optimization);
        config.ortExecutionMode = static_cast<OrtExecutionMode>(source->ort_execution_mode);
        config.ortStrictProvider = source->ort_strict_provider != 0;
        config.deviceId = source->device_id;
        config.intraOpThreads = source->intra_op_threads;
        config.interOpThreads = source->inter_op_threads;
        config.enableMemoryPattern = source->enable_memory_pattern != 0;
        config.enableCpuMemoryArena = source->enable_cpu_memory_arena != 0;
        config.enableProfiling = source->enable_profiling != 0;
        if (source->profile_file_prefix_utf8 && *source->profile_file_prefix_utf8)
            config.profileFilePrefix = pathFromUtf8(source->profile_file_prefix_utf8);

        auto instance = std::make_unique<Instance>();
        instance->backend = makeBackend();
        auto loaded = instance->backend->load(config);
        if (!loaded) return fail(nullptr, loaded.status());

        const auto& signature = instance->backend->signature();
        instance->inputViews.reserve(signature.inputs.size());
        instance->outputViews.reserve(signature.outputs.size());
        for (const auto& spec : signature.inputs) {
            instance->inputViews.push_back({spec.name.c_str(), static_cast<int32_t>(spec.dtype),
                                            spec.shape.dims.data(), spec.shape.dims.size()});
        }
        for (const auto& spec : signature.outputs) {
            instance->outputViews.push_back({spec.name.c_str(), static_cast<int32_t>(spec.dtype),
                                             spec.shape.dims.data(), spec.shape.dims.size()});
        }
        *outInstance = instance.release();
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(nullptr, Status::error(ErrorCode::OutOfMemory,
                                           "Out of memory while creating backend plugin"));
    } catch (const std::exception& error) {
        globalError = error.what();
        return static_cast<int32_t>(ErrorCode::InternalError);
    }
}

void ANOM_PLUGIN_CALL destroy(void* opaque) {
    delete static_cast<Instance*>(opaque);
}

int32_t ANOM_PLUGIN_CALL getSignature(void* opaque, anom_plugin_signature_v1* outSignature) {
    auto* instance = static_cast<Instance*>(opaque);
    if (!instance || !outSignature) {
        return fail(instance, Status::error(ErrorCode::InvalidArgument,
                                            "Backend signature arguments are null"));
    }
    *outSignature = {instance->inputViews.data(), instance->inputViews.size(),
                     instance->outputViews.data(), instance->outputViews.size()};
    return 0;
}

int32_t ANOM_PLUGIN_CALL getMaxBatchSize(void* opaque) {
    auto* instance = static_cast<Instance*>(opaque);
    return instance ? instance->backend->maxBatchSize() : 0;
}

int32_t ANOM_PLUGIN_CALL infer(
    void* opaque, const anom_plugin_tensor_view_v1* inputs, std::size_t inputCount,
    anom_plugin_tensor_batch_v1* outBatch) {
    auto* instance = static_cast<Instance*>(opaque);
    if (!instance || !outBatch || (inputCount != 0 && !inputs)) {
        return fail(instance, Status::error(ErrorCode::InvalidArgument,
                                            "Backend inference arguments are invalid"));
    }
    *outBatch = {};
    try {
        TensorMap converted;
        converted.reserve(inputCount);
        for (std::size_t i = 0; i < inputCount; ++i) {
            const auto& view = inputs[i];
            if (!view.name_utf8 || !validDataType(view.data_type) ||
                (view.rank != 0 && !view.dimensions) ||
                (view.byte_size != 0 && !view.data)) {
                return fail(instance, Status::error(ErrorCode::InvalidArgument,
                                                    "Backend input tensor view is invalid"));
            }
            Tensor tensor;
            tensor.dtype = static_cast<DataType>(view.data_type);
            if (view.rank != 0)
                tensor.shape.dims.assign(view.dimensions, view.dimensions + view.rank);
            tensor.setExternalView(view.data, view.byte_size);
            converted.emplace(view.name_utf8, std::move(tensor));
        }

        auto inferred = instance->backend->infer(converted);
        if (!inferred) return fail(instance, inferred.status());
        auto owner = std::make_unique<BatchOwner>();
        owner->tensors = std::move(inferred.value());
        owner->views.reserve(owner->tensors.size());
        for (const auto& [name, tensor] : owner->tensors) {
            owner->views.push_back({name.c_str(), static_cast<int32_t>(tensor.dtype),
                                    tensor.shape.dims.data(), tensor.shape.dims.size(),
                                    tensor.data<std::byte>(), tensor.byteSize()});
        }
        outBatch->tensors = owner->views.data();
        outBatch->count = owner->views.size();
        outBatch->owner = owner.release();
        instance->lastError.clear();
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(instance, Status::error(ErrorCode::OutOfMemory,
                                            "Out of memory during backend inference"));
    } catch (const std::exception& error) {
        instance->lastError = error.what();
        return static_cast<int32_t>(ErrorCode::InternalError);
    }
}

void ANOM_PLUGIN_CALL releaseBatch(anom_plugin_tensor_batch_v1* batch) {
    if (!batch) return;
    delete static_cast<BatchOwner*>(batch->owner);
    *batch = {};
}

size_t ANOM_PLUGIN_CALL getLastError(void* opaque, char* buffer, size_t bufferSize) {
    const auto* instance = static_cast<const Instance*>(opaque);
    const std::string& message = instance ? instance->lastError : globalError;
    const std::size_t required = message.size() + 1;
    if (buffer && bufferSize != 0) {
        const std::size_t count = std::min(message.size(), bufferSize - 1);
        std::memcpy(buffer, message.data(), count);
        buffer[count] = '\0';
    }
    return required;
}

}  // namespace

extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_backend_plugin_query_v1(
    uint32_t hostAbiVersion, anom_backend_plugin_api_v1* outApi) {
    if (!outApi || hostAbiVersion != ANOM_PLUGIN_ABI_VERSION ||
        outApi->struct_size < sizeof(anom_backend_plugin_api_v1)) {
        return -1;
    }
    anom_backend_plugin_api_v1 api{};
    api.struct_size = sizeof(api);
    api.abi_version = ANOM_PLUGIN_ABI_VERSION;
    api.backend_name_utf8 = ANOM_BACKEND_NAME;
    api.create = &create;
    api.destroy = &destroy;
    api.get_signature = &getSignature;
    api.get_max_batch_size = &getMaxBatchSize;
    api.infer = &infer;
    api.release_batch = &releaseBatch;
    api.get_last_error = &getLastError;
    *outApi = api;
    return 0;
}
