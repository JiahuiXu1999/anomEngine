#pragma once
#include "model/result.h"
#include "plugin_api/anom_faiss_plugin.h"
#include "plugins/dynamic_library.h"

namespace anom::model {
class GpuFaissClient {
public:
    static Result<std::unique_ptr<GpuFaissClient>> load(const std::filesystem::path& index,
        int device, int dimension, int64_t count, int neighbors,
        const std::filesystem::path& pluginDirectory);
    ~GpuFaissClient();
    GpuFaissClient(const GpuFaissClient&) = delete;
    GpuFaissClient& operator=(const GpuFaissClient&) = delete;
    Result<void> search(int64_t count, const float* queries, int neighbors,
                        float* distances, int64_t* labels) const;
private:
    GpuFaissClient() = default;
    std::shared_ptr<plugins::DynamicLibrary> library_;
    anom_faiss_api_v1 api_{};
    void* instance_{nullptr};
};
}
