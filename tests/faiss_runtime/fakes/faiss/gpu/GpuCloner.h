#pragma once
#include "faiss/gpu/GpuIndex.h"
#include "faiss/gpu/StandardGpuResources.h"
namespace faiss::gpu {
struct GpuClonerOptions { bool useFloat16 = false; bool useFloat16CoarseQuantizer = false; };
inline Index* index_cpu_to_gpu(StandardGpuResources*, int device, const Index* cpu, const GpuClonerOptions*) {
    if (cpu->d == 7) throw std::runtime_error("injected unsupported GPU index");
    return new GpuIndex(*cpu, device);
}
}
