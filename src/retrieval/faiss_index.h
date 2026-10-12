#pragma once
#include "retrieval/gpu_faiss_client.h"
#include "retrieval/search_execution.h"
#include <faiss/Index.h>
#include <vector>

namespace anom::model {
class FaissIndex {
public:
    FaissIndex(std::unique_ptr<faiss::Index> cpu, std::filesystem::path path)
        : cpu_(std::move(cpu)), path_(std::move(path)) {}
    int dimension() const { return cpu_->d; }
    Result<void> enableGpu(const SearchExecutionConfig& config, int neighbors);
    bool supportsDevice() const { return gpu_ && gpu_->supportsDevice(); }
    Result<void> searchDevice(int device, int64_t count, const float* queries, int neighbors, float* distances, int64_t* labels) const {
        if (!supportsDevice()) return Status::error(ErrorCode::DeviceUnavailable, "No device search");
        return gpu_->searchDevice(device, count, queries, neighbors, distances, labels);
    }
    void useCpu() { gpu_.reset(); }
    void search(faiss::idx_t count, const float* queries, int neighbors,
                float* distances, faiss::idx_t* labels) const;
    // Weighted PatchCore needs a single support vector. Keep the CPU index for
    // exact reconstruction, including index types with CPU-only reconstruction.
    void reconstruct(faiss::idx_t key, float* output) const { cpu_->reconstruct(key, output); }
private:
    std::unique_ptr<faiss::Index> cpu_;
    std::filesystem::path path_;
    std::unique_ptr<GpuFaissClient> gpu_;
};
Result<SearchExecutionInfo> configureFaissIndexes(const std::vector<FaissIndex*>& indexes,
    const SearchExecutionConfig& config, int neighbors);
}
