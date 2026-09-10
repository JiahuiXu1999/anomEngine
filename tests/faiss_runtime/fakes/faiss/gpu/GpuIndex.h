#pragma once
#include "faiss/Index.h"
#include "cuda_runtime_api.h"
namespace faiss::gpu {
struct GpuIndex : Index {
    int device;
    explicit GpuIndex(const Index& cpu, int selected) : Index(cpu), device(selected) {}
    ~GpuIndex() override { assert(device == fakeDevice); }
    void search(idx_t count, const float* queries, idx_t k, float* distances, idx_t* labels) const override {
        if (device != fakeDevice) throw std::runtime_error("wrong GPU selected");
        if (queries[0] == -999.0F) throw std::runtime_error("injected runtime search failure");
        Index::search(count, queries, k, distances, labels);
    }
};
}
