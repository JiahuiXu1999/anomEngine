#pragma once
#include "cuda_runtime_api.h"
#include <cstddef>
namespace faiss::gpu {
struct StandardGpuResources {
    int device = fakeDevice;
    void setTempMemory(size_t) {}
    cudaStream_t getDefaultStream(int selected) { assert(device == selected); return selected; }
    ~StandardGpuResources() { assert(device == fakeDevice); }
};
}
