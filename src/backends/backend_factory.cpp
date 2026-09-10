#include "backends/backend_factory.h"

#include "infrastructure/utf8_path.h"
#include "plugin_api/anom_backend_plugin.h"
#include "plugins/dynamic_library.h"

#include <cstring>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace anom::model {
namespace {

ErrorCode pluginErrorCode(int32_t value) noexcept {
    if (value >= static_cast<int32_t>(ErrorCode::InvalidArgument) &&
        value <= static_cast<int32_t>(ErrorCode::DeviceUnavailable)) {
        return static_cast<ErrorCode>(value);
    }
    return ErrorCode::BackendFailure;
}

std::string pluginError(const anom_backend_plugin_api_v1& api, void* instance) {
    if (!api.get_last_error) return {};
    const std::size_t required = api.get_last_error(instance, nullptr, 0);
    if (required == 0) return {};
    std::string message(required, '\0');
    api.get_last_error(instance, message.data(), message.size());
    if (!message.empty() && message.back() == '\0') message.pop_back();
    return message;
}

class DynamicBackend final : public IRuntimeBackend {
public:
    DynamicBackend(std::shared_ptr<plugins::DynamicLibrary> library,
                   anom_backend_plugin_api_v1 api, anom_backend_execution_api_v1 execution)
        : library_(std::move(library)), api_(api), execution_(execution) {}

    ~DynamicBackend() override = default;

    Result<void> load(const BackendConfig& config) override {
        if (config.backend == RuntimeBackend::TensorRT && config.provider == ExecutionProvider::Cpu)
            return Status::error(ErrorCode::InvalidArgument, "TensorRT requires the CUDA provider");
        if (instance_) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "Backend plugin is already loaded");
        }
        const std::string onnxPath = pathToUtf8(config.onnxPath);
        const std::string enginePath = pathToUtf8(config.enginePath);
        const std::string profilePath = pathToUtf8(config.profileFilePrefix);
        anom_backend_config_v1 converted{};
        converted.struct_size = sizeof(converted);
        converted.onnx_path_utf8 = onnxPath.c_str();
        converted.engine_path_utf8 = enginePath.c_str();
        converted.load_policy = static_cast<int32_t>(config.loadPolicy);
        converted.fp16 = config.fp16 ? 1 : 0;
        converted.max_batch_size = config.maxBatchSize;
        converted.workspace_bytes = config.workspaceBytes;
        converted.input_name_utf8 = config.inputName.c_str();
        converted.input_layout = static_cast<int32_t>(config.inputLayout);
        converted.input_height = config.inputHeight;
        converted.input_width = config.inputWidth;
        converted.ort_graph_optimization = static_cast<int32_t>(config.ortGraphOptimization);
        converted.ort_execution_mode = static_cast<int32_t>(config.ortExecutionMode);
        converted.device_id = config.deviceId;
        converted.intra_op_threads = config.intraOpThreads;
        converted.inter_op_threads = config.interOpThreads;
        converted.enable_memory_pattern = config.enableMemoryPattern ? 1 : 0;
        converted.enable_cpu_memory_arena = config.enableCpuMemoryArena ? 1 : 0;
        converted.enable_profiling = config.enableProfiling ? 1 : 0;
        converted.profile_file_prefix_utf8 = profilePath.c_str();

        void* instance = nullptr;
        const bool needsExtension = config.provider == ExecutionProvider::Cuda &&
                                    config.backend == RuntimeBackend::OnnxRuntime;
        if (needsExtension && !execution_.create_with_provider)
            return Status::error(ErrorCode::DeviceUnavailable,
                                 "Backend plugin does not support explicit CUDA selection");
        const int32_t created = execution_.create_with_provider
            ? execution_.create_with_provider(&converted, static_cast<int32_t>(config.provider), &instance)
            : api_.create(&converted, &instance);
        if (created != 0 || !instance) {
            const std::string message = pluginError(api_, instance);
            if (instance) api_.destroy(instance);
            return Status::error(pluginErrorCode(created), "Backend plugin creation failed",
                                 message);
        }
        // Output tensors can outlive this wrapper. Keep both the instance and
        // the DLL containing its destroy/release callbacks alive with them.
        instanceLifetime_ = std::shared_ptr<void>(instance,
            [library = library_, destroy = api_.destroy](void* value) {
                destroy(value);
            });
        instance_ = instance;

        anom_plugin_signature_v1 view{};
        const int32_t signatureStatus = api_.get_signature(instance_, &view);
        if (signatureStatus != 0) {
            const std::string message = pluginError(api_, instance_);
            instanceLifetime_.reset();
            instance_ = nullptr;
            return Status::error(pluginErrorCode(signatureStatus),
                                 "Backend plugin signature query failed",
                                 message);
        }
        auto convertedSignature = convertSignature(view);
        if (!convertedSignature) {
            instanceLifetime_.reset();
            instance_ = nullptr;
            return convertedSignature.status();
        }
        signature_ = std::move(convertedSignature.value());
        maxBatchSize_ = api_.get_max_batch_size(instance_);
        if (maxBatchSize_ <= 0) {
            instanceLifetime_.reset();
            instance_ = nullptr;
            return Status::error(ErrorCode::BackendFailure,
                                 "Backend plugin returned an invalid maximum batch size");
        }
        return {};
    }

    Result<void> probe(ExecutionProvider provider, int deviceId) override {
        if (!execution_.probe)
            return Status::error(ErrorCode::DeviceUnavailable,
                                 "Legacy backend plugin does not support runtime probing");
        const auto status = execution_.probe(static_cast<int32_t>(provider), deviceId);
        if (status != 0)
            return Status::error(pluginErrorCode(status), "Runtime is unavailable", pluginError(api_, nullptr));
        return {};
    }

    Result<TensorMap> infer(const TensorMap& inputs) override {
        if (!instance_) {
            return Status::error(ErrorCode::NotInitialized, "Backend plugin is not loaded");
        }
        std::vector<anom_plugin_tensor_view_v1> views;
        views.reserve(inputs.size());
        for (const auto& [name, tensor] : inputs) {
            views.push_back({name.c_str(), static_cast<int32_t>(tensor.dtype),
                             tensor.shape.dims.data(), tensor.shape.dims.size(),
                             tensor.data<std::byte>(), tensor.byteSize()});
        }
        anom_plugin_tensor_batch_v1 batch{};
        const int32_t inferred = api_.infer(instance_, views.data(), views.size(), &batch);
        if (inferred != 0) {
            return Status::error(pluginErrorCode(inferred), "Backend plugin inference failed",
                                 pluginError(api_, instance_));
        }
        if (batch.count != 0 && (!batch.tensors || !batch.owner)) {
            if (batch.owner) api_.release_batch(&batch);
            return Status::error(ErrorCode::BackendFailure,
                                 "Backend plugin returned an invalid tensor batch");
        }

        struct BatchGuard {
            const anom_backend_plugin_api_v1* api;
            anom_plugin_tensor_batch_v1* batch;
            ~BatchGuard() { if (batch->owner) api->release_batch(batch); }
        } guard{&api_, &batch};

        struct SharedBatch {
            std::shared_ptr<void> instanceLifetime;
            decltype(anom_backend_plugin_api_v1::release_batch) release{nullptr};
            anom_plugin_tensor_batch_v1 batch{};
            ~SharedBatch() { if (batch.owner) release(&batch); }
        };
        // Allocate before transferring ownership so an allocation failure still
        // releases the plugin batch through guard.
        auto owner = std::make_shared<SharedBatch>();
        owner->instanceLifetime = instanceLifetime_;
        owner->release = api_.release_batch;
        owner->batch = batch;
        batch = {};

        TensorMap result;
        result.reserve(owner->batch.count);
        for (std::size_t i = 0; i < owner->batch.count; ++i) {
            const auto& view = owner->batch.tensors[i];
            if (!view.name_utf8 || view.data_type < static_cast<int32_t>(DataType::Float32) ||
                view.data_type > static_cast<int32_t>(DataType::Bool) ||
                (view.rank != 0 && !view.dimensions) ||
                (view.byte_size != 0 && !view.data)) {
                return Status::error(ErrorCode::BackendFailure,
                                     "Backend plugin returned an invalid tensor view");
            }
            Tensor tensor;
            tensor.dtype = static_cast<DataType>(view.data_type);
            if (view.rank != 0)
                tensor.shape.dims.assign(view.dimensions, view.dimensions + view.rank);
            tensor.setExternalView(view.data, view.byte_size, owner);
            result.emplace(view.name_utf8, std::move(tensor));
        }
        return result;
    }

    const TensorSignature& signature() const noexcept override { return signature_; }
    int maxBatchSize() const noexcept override { return maxBatchSize_; }

private:
    static Result<TensorSignature> convertSignature(const anom_plugin_signature_v1& source) {
        TensorSignature result;
        const auto append = [](const anom_plugin_tensor_spec_v1* values, std::size_t count,
                               bool input, std::vector<TensorSpec>& destination) -> Result<void> {
            if (count != 0 && !values)
                return Status::error(ErrorCode::BackendFailure,
                                     "Backend plugin signature array is null");
            destination.reserve(count);
            for (std::size_t i = 0; i < count; ++i) {
                const auto& value = values[i];
                if (!value.name_utf8 ||
                    value.data_type < static_cast<int32_t>(DataType::Float32) ||
                    value.data_type > static_cast<int32_t>(DataType::Bool) ||
                    (value.rank != 0 && !value.dimensions)) {
                    return Status::error(ErrorCode::BackendFailure,
                                         "Backend plugin signature is invalid");
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
        auto inputs = append(source.inputs, source.input_count, true, result.inputs);
        if (!inputs) return inputs.status();
        auto outputs = append(source.outputs, source.output_count, false, result.outputs);
        if (!outputs) return outputs.status();
        return result;
    }

    std::shared_ptr<plugins::DynamicLibrary> library_;
    anom_backend_plugin_api_v1 api_{};
    anom_backend_execution_api_v1 execution_{};
    std::shared_ptr<void> instanceLifetime_;
    void* instance_{nullptr};
    TensorSignature signature_;
    int maxBatchSize_{0};
};

}  // namespace

Result<std::unique_ptr<IRuntimeBackend>> createRuntimeBackend(
    RuntimeBackend backend, const std::filesystem::path& pluginDirectory) {
    const char* name = toString(backend);
    const auto directory = pluginDirectory.empty()
        ? plugins::defaultPluginDirectory() : pluginDirectory;
    const auto path = plugins::pluginPath(directory, "backend", name);
    auto library = plugins::DynamicLibrary::open(path);
    if (!library) {
        return Status::error(ErrorCode::PluginNotFound, "Required backend plugin is unavailable",
                             library.status().describe());
    }
    auto query = reinterpret_cast<anom_backend_plugin_query_v1_fn>(
        library.value()->symbol("anom_backend_plugin_query_v1"));
    if (!query) {
        return Status::error(ErrorCode::PluginAbiMismatch,
                             "Backend plugin entry point is missing", pathToUtf8(path));
    }
    anom_backend_plugin_api_v1 api{};
    api.struct_size = sizeof(api);
    const int32_t queried = query(ANOM_PLUGIN_ABI_VERSION, &api);
    if (queried != 0 || api.abi_version != ANOM_PLUGIN_ABI_VERSION ||
        api.struct_size < sizeof(api) || !api.backend_name_utf8 ||
        std::strcmp(api.backend_name_utf8, name) != 0 || !api.create || !api.destroy ||
        !api.get_signature || !api.get_max_batch_size || !api.infer ||
        !api.release_batch || !api.get_last_error) {
        return Status::error(ErrorCode::PluginAbiMismatch,
                             "Backend plugin ABI or function table is invalid", pathToUtf8(path));
    }
    anom_backend_execution_api_v1 execution{};
    auto executionQuery = reinterpret_cast<anom_backend_query_execution_v1_fn>(
        library.value()->symbol("anom_backend_query_execution_v1"));
    if (executionQuery) {
        execution.struct_size = sizeof(execution);
        if (executionQuery(1, &execution) != 0 || execution.abi_version != 1 ||
            execution.struct_size < sizeof(execution) || !execution.create_with_provider || !execution.probe)
            return Status::error(ErrorCode::PluginAbiMismatch, "Backend execution extension is invalid");
    }
    return std::unique_ptr<IRuntimeBackend>(
        new DynamicBackend(std::move(library.value()), api, execution));
}

}  // namespace anom::model
