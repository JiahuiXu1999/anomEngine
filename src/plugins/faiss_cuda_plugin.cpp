#include "plugin_api/anom_faiss_plugin.h"
#include "model/result.h"
#include "infrastructure/utf8_path.h"

#include <faiss/Index.h>
#include <faiss/index_io.h>
#include <faiss/gpu/GpuCloner.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <type_traits>
#include <vector>

namespace {
using namespace anom::model;
thread_local std::string lastError;
int32_t fail(ErrorCode code, const std::string& message) {
    lastError = message;
    return static_cast<int32_t>(code);
}
class DeviceScope {
public:
    explicit DeviceScope(int device) {
        auto status = cudaGetDevice(&previous_);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        status = cudaSetDevice(device);
        if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
        changed_ = previous_ != device;
    }
    ~DeviceScope() { if (changed_) cudaSetDevice(previous_); }
private:
    int previous_{0};
    bool changed_{false};
};
struct Instance {
    int device{0};
    int dimension{0};
    int64_t vectorCount{0};
    std::mutex mutex;
    std::unique_ptr<faiss::gpu::StandardGpuResources> resources;
    std::unique_ptr<faiss::Index> index;
    ~Instance() {
        // In-flight calls must be joined by the owner before destruction.
        try {
            DeviceScope scope(device);
            if (resources) cudaStreamSynchronize(resources->getDefaultStream(device));
            index.reset();
            resources.reset();
        } catch (...) {
            // Device loss is unrecoverable; avoid throwing through the C ABI.
            index.reset();
            resources.reset();
        }
    }
};
struct StreamCompletion {
    cudaStream_t stream;
    ~StreamCompletion() { cudaStreamSynchronize(stream); }
};

int32_t ANOM_PLUGIN_CALL probe(int32_t device) {
    lastError.clear();
    if (device < 0) return fail(ErrorCode::InvalidArgument, "GPU Faiss device id must be non-negative");
    int count = 0;
    const auto status = cudaGetDeviceCount(&count);
    if (status != cudaSuccess || device >= count)
        return fail(ErrorCode::DeviceUnavailable, "Requested GPU Faiss CUDA device is unavailable");
    try { DeviceScope scope(device); return 0; }
    catch (const std::exception& error) { return fail(ErrorCode::DeviceUnavailable, error.what()); }
}

int32_t ANOM_PLUGIN_CALL create(const char* path, int32_t device, int32_t dimension,
                               int64_t vectorCount, int32_t neighbors, void** result) {
    if (!result) return fail(ErrorCode::InvalidArgument, "GPU Faiss output instance is null");
    *result = nullptr;
    if (!path || dimension <= 0 || vectorCount < neighbors || neighbors <= 0)
        return fail(ErrorCode::InvalidArgument, "Invalid GPU Faiss index configuration");
    const auto available = probe(device);
    if (available) return available;
    if (neighbors > 2048)
        return fail(ErrorCode::DeviceUnavailable, "GPU Faiss supports at most 2048 neighbors");
    std::unique_ptr<faiss::Index> cpu;
    try {
        // FILE* avoids narrowing UTF-8 paths through the Windows ANSI codepage.
        const auto native = pathFromUtf8(path);
        FILE* raw = nullptr;
#if defined(_WIN32)
        _wfopen_s(&raw, native.c_str(), L"rb");
#else
        raw = std::fopen(native.c_str(), "rb");
#endif
        if (!raw) return fail(ErrorCode::ArtifactMissing, "Unable to open GPU Faiss index artifact");
        std::unique_ptr<FILE, decltype(&std::fclose)> file(raw, &std::fclose);
        cpu.reset(faiss::read_index(file.get()));
        if (!cpu || cpu->d != dimension || cpu->ntotal != vectorCount || cpu->metric_type != faiss::METRIC_L2)
            return fail(ErrorCode::ArtifactCorrupt, "GPU Faiss index does not match the validated CPU artifact");
    } catch (const std::bad_alloc&) {
        return fail(ErrorCode::OutOfMemory, "Out of memory reading GPU Faiss artifact");
    } catch (const std::exception& error) { return fail(ErrorCode::ArtifactCorrupt, error.what()); }
    try {
        auto instance = std::make_unique<Instance>();
        instance->device = device;
        instance->dimension = dimension;
        instance->vectorCount = vectorCount;
        DeviceScope scope(device);
        instance->resources = std::make_unique<faiss::gpu::StandardGpuResources>();
        // Bound per-index scratch space, especially for multi-layer SPADE.
        instance->resources->setTempMemory(64ULL * 1024ULL * 1024ULL);
        faiss::gpu::GpuClonerOptions options;
        options.useFloat16 = false;
        options.useFloat16CoarseQuantizer = false;
        instance->index.reset(faiss::gpu::index_cpu_to_gpu(instance->resources.get(), device, cpu.get(), &options));
        // Some wrapper index types may keep CPU work after cloning. Require an
        // actual GPU index instead of advertising a CPU/hybrid index as CUDA.
        if (!dynamic_cast<faiss::gpu::GpuIndex*>(instance->index.get()))
            return fail(ErrorCode::DeviceUnavailable, "This Faiss index type cannot run entirely as a GPU index");
        std::vector<float> query(static_cast<size_t>(dimension), 0.0F);
        std::vector<float> distances(static_cast<size_t>(neighbors));
        std::vector<faiss::idx_t> labels(static_cast<size_t>(neighbors));
        StreamCompletion completed{instance->resources->getDefaultStream(device)};
        instance->index->search(1, query.data(), neighbors, distances.data(), labels.data());
        const auto synchronized = cudaStreamSynchronize(completed.stream);
        if (synchronized != cudaSuccess) return fail(ErrorCode::DeviceUnavailable, cudaGetErrorString(synchronized));
        *result = instance.release();
        lastError.clear();
        return 0;
    } catch (const std::bad_alloc&) { return fail(ErrorCode::OutOfMemory, "Out of memory initializing GPU Faiss"); }
    catch (const std::exception& error) {
        // Includes unsupported GPU index types and Faiss CUDA allocation errors.
        return fail(ErrorCode::DeviceUnavailable, error.what());
    }
}
void ANOM_PLUGIN_CALL destroy(void* instance) { delete static_cast<Instance*>(instance); }
int32_t ANOM_PLUGIN_CALL search(void* opaque, int64_t count, const float* queries,
                               int32_t neighbors, float* distances, int64_t* labels) {
    static_assert(std::is_same_v<faiss::idx_t, int64_t>);
    auto* instance = static_cast<Instance*>(opaque);
    if (!instance || count <= 0 || !queries || !distances || !labels || neighbors <= 0 ||
        neighbors > 2048 || neighbors > instance->vectorCount)
        return fail(ErrorCode::InvalidArgument, "Invalid GPU Faiss query");
    const auto limit = std::numeric_limits<size_t>::max() / sizeof(float);
    if (static_cast<uint64_t>(count) > limit / static_cast<size_t>(instance->dimension) ||
        static_cast<uint64_t>(count) > std::numeric_limits<size_t>::max() / sizeof(int64_t) / static_cast<size_t>(neighbors))
        return fail(ErrorCode::InvalidArgument, "GPU Faiss query size overflows address space");
    try {
        std::lock_guard<std::mutex> lock(instance->mutex);
        DeviceScope scope(instance->device);
        StreamCompletion completed{instance->resources->getDefaultStream(instance->device)};
        // Bound temporary query memory while preserving one persistent GPU index.
        for (int64_t begin = 0; begin < count; begin += 4096) {
            const auto batch = std::min<int64_t>(4096, count - begin);
            instance->index->search(batch, queries + begin * instance->dimension, neighbors,
                                   distances + begin * neighbors, labels + begin * neighbors);
        }
        const auto status = cudaStreamSynchronize(completed.stream);
        if (status != cudaSuccess) return fail(ErrorCode::AdapterFailure, cudaGetErrorString(status));
        lastError.clear();
        return 0;
    } catch (const std::exception& error) { return fail(ErrorCode::AdapterFailure, error.what()); }
}
size_t ANOM_PLUGIN_CALL error(void*, char* buffer, size_t size) {
    if (buffer && size) {
        const auto length = std::min(size - 1, lastError.size());
        std::memcpy(buffer, lastError.data(), length);
        buffer[length] = '\0';
    }
    return lastError.size() + 1;
}
}
extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_faiss_query_v1(uint32_t version, anom_faiss_api_v1* api) {
    if (version != 1 || !api || api->struct_size < sizeof(*api)) return -1;
    *api = {sizeof(*api), 1, &probe, &create, &destroy, &search, &error};
    return 0;
}
