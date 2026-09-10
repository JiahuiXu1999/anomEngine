#include "plugin_api/anom_algorithm_plugin.h"

#include "adapters/adapter.h"
#include "infrastructure/utf8_path.h"

#if defined(ANOM_ALGORITHM_PATCHCORE)
#  include "adapters/patchcore_adapter.h"
#elif defined(ANOM_ALGORITHM_PADIM)
#  include "adapters/padim_adapter.h"
#elif defined(ANOM_ALGORITHM_DIRECT)
#  include "adapters/direct_adapter.h"
#elif defined(ANOM_ALGORITHM_EFFICIENTAD)
#  include "adapters/efficientad_adapter.h"
#elif defined(ANOM_ALGORITHM_DFKDE)
#  include "adapters/dfkde_adapter.h"
#elif defined(ANOM_ALGORITHM_SPADE)
#  include "adapters/spade_adapter.h"
#elif defined(ANOM_ALGORITHM_YOLO)
#  include "adapters/yolo_adapter.h"
#else
#  error "An algorithm plugin target must define its algorithm"
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
    std::unique_ptr<IModelAdapter> adapter;
    SearchExecutionInfo searchInfo;
    std::string lastError;
};

struct BatchOwner {
    RawPredictionBatch predictions;
    std::vector<anom_plugin_raw_prediction_v1> views;
};

int32_t fail(Instance* instance, const Status& status) {
    const std::string message = status.describe();
    if (instance) instance->lastError = message;
    else globalError = message;
    return static_cast<int32_t>(status.code);
}

int32_t failInternal(Instance* instance, const char* message) {
    return fail(instance, Status::error(ErrorCode::InternalError, message));
}

std::unique_ptr<IModelAdapter> makeAdapter(const ModelManifest& manifest) {
#if defined(ANOM_ALGORITHM_PATCHCORE)
    return std::make_unique<PatchCoreAdapter>(
        std::get<PatchCoreConfig>(manifest.algorithmConfig), manifest.outputs);
#elif defined(ANOM_ALGORITHM_PADIM)
    return std::make_unique<PadimAdapter>(
        std::get<PadimConfig>(manifest.algorithmConfig), manifest.outputs);
#elif defined(ANOM_ALGORITHM_DIRECT)
    return std::make_unique<DirectPredictionAdapter>(
        std::get<DirectConfig>(manifest.algorithmConfig), manifest.outputs);
#elif defined(ANOM_ALGORITHM_EFFICIENTAD)
    return std::make_unique<EfficientADAdapter>(
        std::get<EfficientADConfig>(manifest.algorithmConfig), manifest.outputs);
#elif defined(ANOM_ALGORITHM_DFKDE)
    return std::make_unique<DfkdeAdapter>(
        std::get<DFKDEConfig>(manifest.algorithmConfig), manifest.outputs);
#elif defined(ANOM_ALGORITHM_SPADE)
    return std::make_unique<SpadeAdapter>(
        std::get<SPADEConfig>(manifest.algorithmConfig), manifest.outputs);
#elif defined(ANOM_ALGORITHM_YOLO)
    return std::make_unique<YoloAdapter>(
        std::get<YoloConfig>(manifest.algorithmConfig), manifest.outputs);
#endif
}

int32_t ANOM_PLUGIN_CALL create(const char* packagePath, void** outInstance) {
    globalError.clear();
    if (!packagePath || !outInstance) {
        return fail(nullptr, Status::error(ErrorCode::InvalidArgument,
                                           "Plugin create arguments are null"));
    }
    *outInstance = nullptr;
    try {
        auto package = ModelPackage::load(pathFromUtf8(packagePath));
        if (!package) return fail(nullptr, package.status());
        if (std::strcmp(toString(package.value().manifest().algorithm),
                        ANOM_ALGORITHM_NAME) != 0) {
            return fail(nullptr, Status::error(ErrorCode::UnsupportedAlgorithm,
                                               "Model package requests a different algorithm",
                                               toString(package.value().manifest().algorithm)));
        }
        auto instance = std::make_unique<Instance>();
        instance->adapter = makeAdapter(package.value().manifest());
        auto loaded = instance->adapter->loadAssets(package.value());
        if (!loaded) return fail(nullptr, loaded.status());
        *outInstance = instance.release();
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(nullptr, Status::error(ErrorCode::OutOfMemory,
                                           "Out of memory while creating algorithm plugin"));
    } catch (const std::exception& error) {
        globalError = error.what();
        return static_cast<int32_t>(ErrorCode::InternalError);
    }
}

void ANOM_PLUGIN_CALL destroy(void* opaque) {
    delete static_cast<Instance*>(opaque);
}

bool validDataType(int32_t type) {
    return type >= static_cast<int32_t>(DataType::Float32) &&
           type <= static_cast<int32_t>(DataType::Bool);
}

int32_t ANOM_PLUGIN_CALL validateSignature(
    void* opaque, const anom_plugin_signature_v1* signature) {
    auto* instance = static_cast<Instance*>(opaque);
    if (!instance || !signature) {
        return fail(instance, Status::error(ErrorCode::InvalidArgument,
                                            "Plugin signature arguments are null"));
    }
    try {
        TensorSignature converted;
        const auto append = [&](const anom_plugin_tensor_spec_v1* values, std::size_t count,
                                bool input, std::vector<TensorSpec>& destination) -> Result<void> {
            if (count != 0 && !values) {
                return Status::error(ErrorCode::InvalidArgument, "Tensor specification array is null");
            }
            destination.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const auto& value = values[i];
                if (!value.name_utf8 || !validDataType(value.data_type) ||
                    (value.rank != 0 && !value.dimensions)) {
                    return Status::error(ErrorCode::InvalidArgument,
                                         "Tensor specification is invalid");
                }
                TensorSpec spec;
                spec.name = value.name_utf8;
                spec.dtype = static_cast<DataType>(value.data_type);
                if (value.rank != 0)
                    spec.shape.dims.assign(value.dimensions, value.dimensions + value.rank);
                spec.input = input;
                destination.push_back(std::move(spec));
            }
            return {};
        };
        auto inputs = append(signature->inputs, signature->input_count, true, converted.inputs);
        if (!inputs) return fail(instance, inputs.status());
        auto outputs = append(signature->outputs, signature->output_count, false, converted.outputs);
        if (!outputs) return fail(instance, outputs.status());
        auto result = instance->adapter->validateSignature(converted);
        if (!result) return fail(instance, result.status());
        instance->lastError.clear();
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(instance, Status::error(ErrorCode::OutOfMemory,
                                            "Out of memory while validating signature"));
    } catch (const std::exception& error) {
        instance->lastError = error.what();
        return static_cast<int32_t>(ErrorCode::InternalError);
    }
}

int32_t ANOM_PLUGIN_CALL predict(
    void* opaque, const anom_plugin_tensor_view_v1* outputs, std::size_t outputCount,
    int32_t inputWidth, int32_t inputHeight, anom_plugin_prediction_batch_v1* outBatch) {
    auto* instance = static_cast<Instance*>(opaque);
    if (!instance || !outBatch || (outputCount != 0 && !outputs) ||
        inputWidth <= 0 || inputHeight <= 0) {
        return fail(instance, Status::error(ErrorCode::InvalidArgument,
                                            "Plugin prediction arguments are invalid"));
    }
    *outBatch = {};
    try {
        TensorMap converted;
        converted.reserve(outputCount);
        for (std::size_t i = 0; i < outputCount; ++i) {
            const auto& view = outputs[i];
            if (!view.name_utf8 || !validDataType(view.data_type) ||
                (view.rank != 0 && !view.dimensions) ||
                (view.byte_size != 0 && !view.data)) {
                return fail(instance, Status::error(ErrorCode::InvalidArgument,
                                                    "Plugin tensor view is invalid"));
            }
            Tensor tensor;
            tensor.dtype = static_cast<DataType>(view.data_type);
            if (view.rank != 0)
                tensor.shape.dims.assign(view.dimensions, view.dimensions + view.rank);
            tensor.setExternalView(view.data, view.byte_size);
            converted.emplace(view.name_utf8, std::move(tensor));
        }
        auto predicted = instance->adapter->predict(converted, cv::Size(inputWidth, inputHeight));
        if (!predicted) return fail(instance, predicted.status());

        auto owner = std::make_unique<BatchOwner>();
        owner->predictions = std::move(predicted.value());
        owner->views.reserve(owner->predictions.size());
        for (const auto& prediction : owner->predictions) {
            anom_plugin_raw_prediction_v1 view{};
            view.score = prediction.score;
            if (!prediction.anomalyMap.empty()) {
                view.anomaly_map = prediction.anomalyMap.ptr<float>();
                view.map_width = prediction.anomalyMap.cols;
                view.map_height = prediction.anomalyMap.rows;
                view.map_stride_elements = static_cast<int32_t>(prediction.anomalyMap.step1());
            }
            owner->views.push_back(view);
        }
        outBatch->predictions = owner->views.data();
        outBatch->count = owner->views.size();
        outBatch->owner = owner.release();
        instance->lastError.clear();
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(instance, Status::error(ErrorCode::OutOfMemory,
                                            "Out of memory during algorithm inference"));
    } catch (const std::exception& error) {
        instance->lastError = error.what();
        return static_cast<int32_t>(ErrorCode::InternalError);
    }
}

void ANOM_PLUGIN_CALL releaseBatch(anom_plugin_prediction_batch_v1* batch) {
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

int32_t ANOM_PLUGIN_CALL configureSearch(void* opaque, const anom_search_config_v1* config,
                                        anom_search_info_v1* info) {
    auto* instance = static_cast<Instance*>(opaque);
    if (!instance || !config || !info || config->struct_size < sizeof(*config) ||
        info->struct_size < sizeof(*info) || (config->provider != 1 && config->provider != 2) ||
        config->device_id < 0)
        return fail(instance, Status::error(ErrorCode::InvalidArgument, "Invalid search configuration"));
    try {
        SearchExecutionConfig converted;
        converted.provider = static_cast<ExecutionProvider>(config->provider);
        converted.deviceId = config->device_id;
        converted.allowCpuFallback = config->allow_cpu_fallback != 0;
        if (config->plugin_directory_utf8) converted.pluginDirectory = pathFromUtf8(config->plugin_directory_utf8);
        auto configured = instance->adapter->configureSearch(converted);
        if (!configured) return fail(instance, configured.status());
        instance->searchInfo = std::move(configured.value());
        const auto& selected = instance->searchInfo;
        *info = {sizeof(*info), static_cast<int32_t>(selected.provider), selected.deviceId,
                 selected.fallbackOccurred ? 1 : 0, selected.fallbackReason.c_str()};
        instance->lastError.clear();
        return 0;
    } catch (const std::bad_alloc&) {
        return fail(instance, Status::error(ErrorCode::OutOfMemory, "Out of memory configuring Faiss"));
    } catch (const std::exception& error) {
        return fail(instance, Status::error(ErrorCode::InternalError, "Faiss configuration failed", error.what()));
    }
}
}  // namespace

extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_algorithm_query_execution_v1(
    uint32_t version, anom_algorithm_execution_api_v1* api) {
    if (version != 1 || !api || api->struct_size < sizeof(*api)) return -1;
    *api = {sizeof(*api), 1, &configureSearch};
    return 0;
}

extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_algorithm_plugin_query_v1(
    uint32_t hostAbiVersion, anom_algorithm_plugin_api_v1* outApi) {
    if (!outApi || hostAbiVersion != ANOM_PLUGIN_ABI_VERSION ||
        outApi->struct_size < sizeof(anom_algorithm_plugin_api_v1)) {
        return -1;
    }
    anom_algorithm_plugin_api_v1 api{};
    api.struct_size = sizeof(api);
    api.abi_version = ANOM_PLUGIN_ABI_VERSION;
    api.algorithm_name_utf8 = ANOM_ALGORITHM_NAME;
    api.create = &create;
    api.destroy = &destroy;
    api.validate_signature = &validateSignature;
    api.predict = &predict;
    api.release_batch = &releaseBatch;
    api.get_last_error = &getLastError;
    *outApi = api;
    return 0;
}
