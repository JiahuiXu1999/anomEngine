#include "adapters/adapter.h"

#include "infrastructure/utf8_path.h"
#include "plugin_api/anom_algorithm_plugin.h"
#include "plugins/dynamic_library.h"

#include <cstring>
#include <utility>
#include <vector>

namespace anom::model {
namespace {

ErrorCode pluginErrorCode(int32_t value) noexcept {
    if (value >= static_cast<int32_t>(ErrorCode::InvalidArgument) &&
        value <= static_cast<int32_t>(ErrorCode::InternalError)) {
        return static_cast<ErrorCode>(value);
    }
    return ErrorCode::AdapterFailure;
}

std::string pluginError(const anom_algorithm_plugin_api_v1& api, void* instance) {
    if (!api.get_last_error) return {};
    const std::size_t required = api.get_last_error(instance, nullptr, 0);
    if (required == 0) return {};
    std::string message(required, '\0');
    api.get_last_error(instance, message.data(), message.size());
    if (!message.empty() && message.back() == '\0') message.pop_back();
    return message;
}

class DynamicAdapter final : public IModelAdapter {
public:
    DynamicAdapter(std::shared_ptr<plugins::DynamicLibrary> library,
                   anom_algorithm_plugin_api_v1 api, void* instance)
        : library_(std::move(library)), api_(api), instance_(instance) {}

    ~DynamicAdapter() override {
        if (instance_ && api_.destroy) api_.destroy(instance_);
    }

    Result<void> loadAssets(const ModelPackage&) override { return {}; }

    Result<void> validateSignature(const TensorSignature& signature) const override {
        std::vector<anom_plugin_tensor_spec_v1> inputs;
        std::vector<anom_plugin_tensor_spec_v1> outputs;
        inputs.reserve(signature.inputs.size());
        outputs.reserve(signature.outputs.size());
        for (const auto& item : signature.inputs) {
            inputs.push_back({item.name.c_str(), static_cast<int32_t>(item.dtype),
                              item.shape.dims.data(), item.shape.dims.size()});
        }
        for (const auto& item : signature.outputs) {
            outputs.push_back({item.name.c_str(), static_cast<int32_t>(item.dtype),
                               item.shape.dims.data(), item.shape.dims.size()});
        }
        const anom_plugin_signature_v1 view{
            inputs.data(), inputs.size(), outputs.data(), outputs.size()};
        const int32_t status = api_.validate_signature(instance_, &view);
        if (status != 0) {
            return Status::error(pluginErrorCode(status), "Algorithm plugin rejected tensor signature",
                                 pluginError(api_, instance_));
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override {
        std::vector<anom_plugin_tensor_view_v1> views;
        views.reserve(outputs.size());
        for (const auto& [name, tensor] : outputs) {
            views.push_back({name.c_str(), static_cast<int32_t>(tensor.dtype),
                             tensor.shape.dims.data(), tensor.shape.dims.size(),
                             tensor.data<std::byte>(), tensor.byteSize()});
        }

        anom_plugin_prediction_batch_v1 batch{};
        const int32_t status = api_.predict(
            instance_, views.data(), views.size(), modelInputSize.width,
            modelInputSize.height, &batch);
        if (status != 0) {
            return Status::error(pluginErrorCode(status), "Algorithm plugin inference failed",
                                 pluginError(api_, instance_));
        }

        struct BatchGuard {
            const anom_algorithm_plugin_api_v1* api;
            anom_plugin_prediction_batch_v1* batch;
            ~BatchGuard() { if (batch->owner) api->release_batch(batch); }
        } guard{&api_, &batch};

        if (batch.count != 0 && !batch.predictions) {
            return Status::error(ErrorCode::AdapterFailure,
                                 "Algorithm plugin returned a null prediction array");
        }
        RawPredictionBatch result(batch.count);
        for (std::size_t i = 0; i < batch.count; ++i) {
            const auto& source = batch.predictions[i];
            auto& destination = result[i];
            destination.score = source.score;
            if (source.map_width == 0 && source.map_height == 0) continue;
            if (!source.anomaly_map || source.map_width <= 0 || source.map_height <= 0 ||
                source.map_stride_elements < source.map_width) {
                return Status::error(ErrorCode::AdapterFailure,
                                     "Algorithm plugin returned an invalid anomaly map");
            }
            cv::Mat map(source.map_height, source.map_width, CV_32FC1,
                        const_cast<float*>(source.anomaly_map),
                        static_cast<std::size_t>(source.map_stride_elements) * sizeof(float));
            destination.anomalyMap = map.clone();
        }
        return result;
    }

private:
    std::shared_ptr<plugins::DynamicLibrary> library_;
    anom_algorithm_plugin_api_v1 api_{};
    void* instance_{nullptr};
};

}  // namespace

Result<std::unique_ptr<IModelAdapter>> createAdapter(
    const ModelPackage& package, const std::filesystem::path& pluginDirectory) {
    const char* algorithm = toString(package.manifest().algorithm);
    const auto directory = pluginDirectory.empty()
        ? plugins::defaultPluginDirectory() : pluginDirectory;
    const auto path = plugins::pluginPath(directory, "algo", algorithm);
    auto library = plugins::DynamicLibrary::open(path);
    if (!library) {
        return Status::error(ErrorCode::UnsupportedAlgorithm,
                             "Required algorithm plugin is unavailable",
                             library.status().describe());
    }

    auto query = reinterpret_cast<anom_algorithm_plugin_query_v1_fn>(
        library.value()->symbol("anom_algorithm_plugin_query_v1"));
    if (!query) {
        return Status::error(ErrorCode::UnsupportedAlgorithm,
                             "Algorithm plugin entry point is missing", pathToUtf8(path));
    }

    anom_algorithm_plugin_api_v1 api{};
    api.struct_size = sizeof(api);
    const int32_t queried = query(ANOM_PLUGIN_ABI_VERSION, &api);
    if (queried != 0 || api.abi_version != ANOM_PLUGIN_ABI_VERSION ||
        api.struct_size < sizeof(api)) {
        return Status::error(ErrorCode::UnsupportedAlgorithm,
                             "Algorithm plugin ABI is incompatible", pathToUtf8(path));
    }
    if (!api.algorithm_name_utf8 || std::strcmp(api.algorithm_name_utf8, algorithm) != 0 ||
        !api.create || !api.destroy || !api.validate_signature || !api.predict ||
        !api.release_batch || !api.get_last_error) {
        return Status::error(ErrorCode::UnsupportedAlgorithm,
                             "Algorithm plugin function table is invalid", pathToUtf8(path));
    }

    void* instance = nullptr;
    const std::string packagePath = pathToUtf8(package.root());
    const int32_t created = api.create(packagePath.c_str(), &instance);
    if (created != 0 || !instance) {
        const std::string message = pluginError(api, instance);
        if (instance) api.destroy(instance);
        return Status::error(pluginErrorCode(created), "Algorithm plugin creation failed",
                             message);
    }
    return std::unique_ptr<IModelAdapter>(
        new DynamicAdapter(std::move(library.value()), api, instance));
}

}  // namespace anom::model
