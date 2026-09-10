#include "retrieval/gpu_faiss_client.h"
#include "infrastructure/utf8_path.h"
#include "model/runtime_types.h"

namespace anom::model {
namespace {
Status failure(const anom_faiss_api_v1& api, void* instance, int32_t code) {
    const size_t size = api.get_last_error(instance, nullptr, 0);
    std::string message(size, '\0');
    if (size) api.get_last_error(instance, message.data(), size);
    if (!message.empty() && message.back() == '\0') message.pop_back();
    const auto mapped = code >= static_cast<int32_t>(ErrorCode::InvalidArgument) &&
        code <= static_cast<int32_t>(ErrorCode::DeviceUnavailable)
        ? static_cast<ErrorCode>(code) : ErrorCode::AdapterFailure;
    return Status::error(mapped, "GPU Faiss operation failed", message);
}
}
Result<std::unique_ptr<GpuFaissClient>> GpuFaissClient::load(const std::filesystem::path& index,
    int device, int dimension, int64_t count, int neighbors, const std::filesystem::path& directory) {
    if (device < 0 || dimension <= 0 || neighbors <= 0 || count < neighbors)
        return Status::error(ErrorCode::InvalidArgument, "Invalid GPU Faiss configuration");
    auto client = std::unique_ptr<GpuFaissClient>(new GpuFaissClient);
    const auto path = plugins::pluginPath(directory.empty() ? plugins::defaultPluginDirectory() : directory,
                                          "search", "faiss_cuda");
    auto library = plugins::DynamicLibrary::open(path);
    if (!library) return Status::error(ErrorCode::PluginNotFound, "GPU Faiss plugin is unavailable", library.status().describe());
    client->library_ = std::move(library.value());
    auto query = reinterpret_cast<anom_faiss_query_v1_fn>(client->library_->symbol("anom_faiss_query_v1"));
    auto& api = client->api_;
    api.struct_size = sizeof(api);
    if (!query || query(1, &api) != 0 || api.abi_version != 1 || api.struct_size < sizeof(api) ||
        !api.probe || !api.create || !api.destroy || !api.search || !api.get_last_error)
        return Status::error(ErrorCode::PluginAbiMismatch, "Invalid GPU Faiss plugin ABI");
    auto status = api.probe(device);
    if (status) return failure(api, nullptr, status);
    const auto encoded = pathToUtf8(index);
    status = api.create(encoded.c_str(), device, dimension, count, neighbors, &client->instance_);
    if (status) return failure(api, client->instance_, status);
    if (!client->instance_) return Status::error(ErrorCode::AdapterFailure, "GPU Faiss returned a null instance");
    return client;
}
GpuFaissClient::~GpuFaissClient() { if (instance_) api_.destroy(instance_); }
Result<void> GpuFaissClient::search(int64_t count, const float* queries, int neighbors,
                                   float* distances, int64_t* labels) const {
    const auto status = api_.search(instance_, count, queries, neighbors, distances, labels);
    if (status) return failure(api_, instance_, status);
    return {};
}
}
