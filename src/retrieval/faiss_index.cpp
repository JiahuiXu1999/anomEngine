#include "retrieval/faiss_index.h"
#include <stdexcept>
#include <type_traits>

namespace anom::model {
Result<void> FaissIndex::enableGpu(const SearchExecutionConfig& config, int neighbors) {
    auto gpu = GpuFaissClient::load(path_, config.deviceId, cpu_->d, cpu_->ntotal, neighbors, config.pluginDirectory);
    if (!gpu) return gpu.status();
    gpu_ = std::move(gpu.value());
    return {};
}
void FaissIndex::search(faiss::idx_t count, const float* queries, int neighbors,
                        float* distances, faiss::idx_t* labels) const {
    if (gpu_) {
        static_assert(std::is_same_v<faiss::idx_t, int64_t>);
        auto searched = gpu_->search(count, queries, neighbors, distances, labels);
        if (!searched) throw std::runtime_error(searched.status().describe());
    } else cpu_->search(count, queries, neighbors, distances, labels);
}
Result<SearchExecutionInfo> configureFaissIndexes(const std::vector<FaissIndex*>& indexes,
    const SearchExecutionConfig& config, int neighbors) {
    if (indexes.empty()) return Status::error(ErrorCode::NotInitialized, "Faiss indexes are not loaded");
    if (config.provider != ExecutionProvider::Cpu && config.provider != ExecutionProvider::Cuda)
        return Status::error(ErrorCode::InvalidArgument, "Invalid Faiss execution provider");
    for (auto* index : indexes) index->useCpu();
    SearchExecutionInfo info;
    info.provider = ExecutionProvider::Cpu;
    if (config.provider == ExecutionProvider::Cpu) return info;
    for (auto* index : indexes) {
        auto selected = index->enableGpu(config, neighbors);
        if (!selected) {
            // All layers must use one provider; release any already-cloned GPU indexes.
            for (auto* loaded : indexes) loaded->useCpu();
            if (!config.allowCpuFallback || !retryableSearchFailure(selected.status().code)) return selected.status();
            info.fallbackOccurred = true;
            info.fallbackReason = selected.status().describe();
            return info;
        }
    }
    info.provider = ExecutionProvider::Cuda;
    info.deviceId = config.deviceId;
    return info;
}
}
