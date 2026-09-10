#pragma once
#include <cassert>
using cudaError_t = int;
using cudaStream_t = int;
constexpr cudaError_t cudaSuccess = 0;
inline thread_local int fakeDevice = 0;
inline cudaError_t cudaGetDevice(int* device) { *device = fakeDevice; return 0; }
inline cudaError_t cudaSetDevice(int device) { if (device < 0 || device > 1) return 1; fakeDevice = device; return 0; }
inline cudaError_t cudaGetDeviceCount(int* count) { *count = 2; return 0; }
inline const char* cudaGetErrorString(cudaError_t) { return "fake CUDA error"; }
inline cudaError_t cudaStreamSynchronize(cudaStream_t stream) { assert(stream == fakeDevice); (void)stream; return 0; }
