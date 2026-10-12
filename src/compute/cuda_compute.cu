#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <math_constants.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#include "model/result.h"
#include "plugin_api/anom_compute_plugin.h"

namespace {
using anom::model::ErrorCode;
thread_local std::string lastError;
void check(cudaError_t s) {
    if (s != cudaSuccess) throw std::runtime_error(cudaGetErrorString(s));
}
void blas(cublasStatus_t s) {
    if (s != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS failure: " + std::to_string(s));
}
void solver(cusolverStatus_t s) {
    if (s != CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER failure: " + std::to_string(s));
}
void require(bool b, const char* text) {
    if (!b) throw std::invalid_argument(text);
}
size_t elements(int a, int b = 1, int c = 1, int d = 1) {
    size_t n = 1;
    for (int v : {a, b, c, d}) {
        require(v > 0, "Dimensions must be positive");
        require(n <= SIZE_MAX / size_t(v), "Size overflow");
        n *= v;
    }
    require(n <= SIZE_MAX / sizeof(double), "Byte size overflow");
    return n;
}
struct DeviceScope {
    int previous;
    explicit DeviceScope(int d) {
        check(cudaGetDevice(&previous));
        check(cudaSetDevice(d));
    }
    ~DeviceScope() { cudaSetDevice(previous); }
};
struct Context {
    int device;
    cudaStream_t stream{};
    cublasHandle_t handle{};
    cusolverDnHandle_t eigen{};
    std::mutex mutex;
    explicit Context(int d) : device(d) {}
    void init() {
        DeviceScope scope(device);
        check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        blas(cublasCreate(&handle));
        blas(cublasSetStream(handle, stream));
        blas(cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH));
        solver(cusolverDnCreate(&eigen));
        solver(cusolverDnSetStream(eigen, stream));
    }
    ~Context() {
        try {
            DeviceScope scope(device);
            if (stream) cudaStreamSynchronize(stream);
            if (eigen) cusolverDnDestroy(eigen);
            if (handle) cublasDestroy(handle);
            if (stream) cudaStreamDestroy(stream);
        } catch (...) {
        }
    }
};
struct Completion {
    cudaStream_t stream;
    ~Completion() { cudaStreamSynchronize(stream); }
};
template <class T>
struct Buffer {
    T* p{};
    explicit Buffer(size_t n) { check(cudaMalloc(reinterpret_cast<void**>(&p), n * sizeof(T))); }
    ~Buffer() { cudaFree(p); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
};
struct Argument {
    void* p{};
    bool owned{};
    anom_compute_buffer b{};
    Context& ctx;
    Argument(Context& c, anom_compute_buffer x, size_t bytes, bool output = false) : b(x), ctx(c) {
        require(x.data && x.bytes >= bytes, "Null or undersized buffer");
        require(x.device == -1 || x.device == c.device, "Buffer belongs to another device");
        if (x.device == -1) {
            check(cudaMalloc(&p, bytes));
            owned = true;
            if (!output) {
                auto s = cudaMemcpyAsync(p, x.data, bytes, cudaMemcpyHostToDevice, c.stream);
                if (s != cudaSuccess) {
                    cudaFree(p);
                    p = nullptr;
                    check(s);
                }
            }
        } else
            p = x.data;
    }
    ~Argument() {
        if (owned) cudaFree(p);
    }
    template <class T>
    T* as() {
        return static_cast<T*>(p);
    }
    void finish(size_t bytes) {
        if (owned) check(cudaMemcpyAsync(b.data, p, bytes, cudaMemcpyDeviceToHost, ctx.stream));
    }
};
int blocks(size_t n) { return int(std::min<size_t>((n + 255) / 256, 65535)); }
#include "compute/cuda_kernels.cuh"
void validateFinite(Context& c, const float* x, size_t n) {
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, sizeof(int), c.stream));
    finiteFloats<<<blocks(n), 256, 0, c.stream>>>(x, n, invalid.p);
    int bad = 0;
    check(cudaMemcpyAsync(&bad, invalid.p, sizeof(int), cudaMemcpyDeviceToHost, c.stream));
    check(cudaStreamSynchronize(c.stream));
    require(!bad, "Non-finite compute features");
}
int pooledSize(int input, int kernel, int stride, int padding) {
    require(input > 0 && kernel > 0 && kernel <= 1024 && stride > 0 && padding >= 0 && padding <= 1024,
            "Invalid pooling dimensions");
    const int64_t padded = int64_t(input) + 2LL * padding;
    require(padded >= kernel, "Empty pooled grid");
    const int64_t output = (padded - kernel) / stride + 1;
    require(output <= INT_MAX, "Pooled dimension overflow");
    return static_cast<int>(output);
}
void runAggregate(Context& c, const anom_compute_aggregate& a) {
    require(a.layers && a.count > 0 && a.count <= 1024, "Invalid layers");
    require(a.kernel > 0 && a.kernel <= 1024 && a.stride > 0 && a.padding >= 0, "Invalid pooling");
    const auto& first = a.layers[0];
    int oh = pooledSize(first.h, a.kernel, a.stride, a.padding),
        ow = pooledSize(first.w, a.kernel, a.stride, a.padding), dimension = 0;
    for (size_t j = 0; j < a.count; ++j) {
        auto l = a.layers[j];
        elements(l.n, l.c, l.h, l.w);
        require(l.n == first.n, "Batch mismatch");
        require(l.h + 2LL * a.padding >= a.kernel && l.w + 2LL * a.padding >= a.kernel, "Empty pooled layer");
        if (a.mean) require(l.c == first.c && l.h == first.h && l.w == first.w, "Mean shape mismatch");
        require(l.c <= INT_MAX - dimension, "Channel overflow");
        dimension += l.c;
    }
    if (a.mean) dimension = first.c;
    size_t total = elements(first.n, oh, ow, dimension);
    Argument out(c, a.output, total * sizeof(float), true);
    if (a.mean) check(cudaMemsetAsync(out.p, 0, total * sizeof(float), c.stream));
    int offset = 0;
    for (size_t j = 0; j < a.count; ++j) {
        auto l = a.layers[j];
        int ph = pooledSize(l.h, a.kernel, a.stride, a.padding),
            pw = pooledSize(l.w, a.kernel, a.stride, a.padding);
        Argument in(c, l.buffer, elements(l.n, l.c, l.h, l.w) * sizeof(float));
        size_t pn = elements(l.n, l.c, ph, pw);
        Buffer<float> pooled(pn);
        pool<<<blocks(pn), 256, 0, c.stream>>>(in.as<float>(), pooled.p, pn, l.h, l.w, ph, pw, a.kernel,
                                               a.stride, a.padding);
        size_t sn = elements(l.n, l.c, oh, ow);
        pack<<<blocks(sn), 256, 0, c.stream>>>(pooled.p, out.as<float>(), sn, l.c, ph, pw, oh, ow, dimension,
                                               offset, a.mean, int(a.count));
        if (!a.mean) offset += l.c;
    }
    if (a.mean) divide<<<blocks(total), 256, 0, c.stream>>>(out.as<float>(), total, float(a.count));
    out.finish(total * sizeof(float));
}
void runProject(Context& c, const anom_compute_project& a) {
    size_t n = elements(a.rows, a.dimension), m = elements(a.dimension, a.rank), o = elements(a.rows, a.rank);
    Argument x(c, a.input, n * 4), r(c, a.matrix, m * 4), y(c, a.output, o * 4, true);
    float one = 1, zero = 0;
    blas(cublasSgemm(c.handle, CUBLAS_OP_N, CUBLAS_OP_N, a.rank, a.rows, a.dimension, &one, r.as<float>(),
                     a.rank, x.as<float>(), a.dimension, &zero, y.as<float>(), a.rank));
    y.finish(o * 4);
}
void runCoreset(Context& c, const anom_compute_coreset& a) {
    auto n = elements(a.rows, a.dimension);
    require(a.count > 0 && a.count <= a.rows && a.first >= 0 && a.first < a.rows && a.selected,
            "Invalid coreset selection");
    Argument x(c, a.input, n * 4);
    validateFinite(c, x.as<float>(), n);
    Buffer<double> minimum(a.rows);
    Buffer<int> center(1);
    Buffer<int64_t> selected(a.count);
    int groups = blocks(a.rows);
    Buffer<Best> partial(groups);
    check(cudaMemcpyAsync(center.p, &a.first, sizeof(int), cudaMemcpyHostToDevice, c.stream));
    int64_t first = a.first;
    check(cudaMemcpyAsync(selected.p, &first, 8, cudaMemcpyHostToDevice, c.stream));
    initDistances<<<groups, 256, 0, c.stream>>>(minimum.p, a.rows);
    for (int i = 1; i < a.count; ++i) {
        distances<<<groups, 256, 0, c.stream>>>(x.as<float>(), minimum.p, partial.p, center.p, a.rows,
                                                a.dimension);
        nextCenter<<<1, 256, 0, c.stream>>>(partial.p, groups, center.p, selected.p, i);
    }
    check(cudaMemcpyAsync(a.selected, selected.p, size_t(a.count) * 8, cudaMemcpyDeviceToHost, c.stream));
}
void runTransform(Context& c, const anom_compute_transform& a) {
    size_t n = elements(a.rows, a.dimension), nr = elements(a.rows, a.rank);
    require(a.rank <= a.dimension && a.mean && a.basis && (a.residual || a.scales), "Invalid statistics");
    Argument x(c, a.input, n * 4), y(c, a.output, (a.residual ? size_t(a.rows) : nr) * 4, true);
    Buffer<double> mean(a.dimension), basis(elements(a.rank, a.dimension)), centered(n), projected(nr);
    check(cudaMemcpyAsync(mean.p, a.mean, size_t(a.dimension) * 8, cudaMemcpyHostToDevice, c.stream));
    check(cudaMemcpyAsync(basis.p, a.basis, size_t(a.rank) * a.dimension * 8, cudaMemcpyHostToDevice,
                          c.stream));
    centerData<<<blocks(n), 256, 0, c.stream>>>(x.as<float>(), centered.p, mean.p, n, a.dimension);
    double one = 1, zero = 0;
    blas(cublasDgemm(c.handle, CUBLAS_OP_T, CUBLAS_OP_N, a.rank, a.rows, a.dimension, &one, basis.p,
                     a.dimension, centered.p, a.dimension, &zero, projected.p, a.rank));
    if (a.residual) {
        Buffer<double> reconstruction(n);
        blas(cublasDgemm(c.handle, CUBLAS_OP_N, CUBLAS_OP_N, a.dimension, a.rows, a.rank, &one, basis.p,
                         a.dimension, projected.p, a.rank, &zero, reconstruction.p, a.dimension));
        residualData<<<blocks(a.rows), 256, 0, c.stream>>>(centered.p, reconstruction.p, y.as<float>(),
                                                           a.rows, a.dimension);
    } else {
        Buffer<double> scale(a.rank);
        check(cudaMemcpyAsync(scale.p, a.scales, size_t(a.rank) * 8, cudaMemcpyHostToDevice, c.stream));
        scaleData<<<blocks(nr), 256, 0, c.stream>>>(projected.p, y.as<float>(), scale.p, nr, a.rank);
    }
    validateFinite(c, y.as<float>(), a.residual ? size_t(a.rows) : nr);
    y.finish((a.residual ? size_t(a.rows) : nr) * 4);
}
void runMoments(Context& c, const anom_compute_moments& a) {
    size_t n = elements(a.rows, a.dimension), dd = elements(a.dimension, a.dimension);
    require(a.mean && a.m2, "Missing moment outputs");
    Argument x(c, a.input, n * 4);
    Buffer<double> mean(a.dimension), centered(n), m2(dd);
    means<<<blocks(a.dimension), 256, 0, c.stream>>>(x.as<float>(), mean.p, a.rows, a.dimension);
    centerData<<<blocks(n), 256, 0, c.stream>>>(x.as<float>(), centered.p, mean.p, n, a.dimension);
    double one = 1, zero = 0;
    blas(cublasDgemm(c.handle, CUBLAS_OP_N, CUBLAS_OP_T, a.dimension, a.dimension, a.rows, &one, centered.p,
                     a.dimension, centered.p, a.dimension, &zero, m2.p, a.dimension));
    check(cudaMemcpyAsync(a.mean, mean.p, size_t(a.dimension) * 8, cudaMemcpyDeviceToHost, c.stream));
    check(cudaMemcpyAsync(a.m2, m2.p, dd * 8, cudaMemcpyDeviceToHost, c.stream));
}
void runEigen(Context& c, const anom_compute_eigen& a) {
    size_t dd = elements(a.dimension, a.dimension);
    require(a.covariance && a.values && a.vectors, "Missing eigen buffers");
    Buffer<double> matrix(dd), values(a.dimension);
    Buffer<int> info(1);
    check(cudaMemcpyAsync(matrix.p, a.covariance, dd * 8, cudaMemcpyHostToDevice, c.stream));
    int workSize = 0;
    solver(cusolverDnDsyevd_bufferSize(c.eigen, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, a.dimension,
                                       matrix.p, a.dimension, values.p, &workSize));
    Buffer<double> work(workSize);
    solver(cusolverDnDsyevd(c.eigen, CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER, a.dimension, matrix.p,
                            a.dimension, values.p, work.p, workSize, info.p));
    int status;
    check(cudaMemcpyAsync(&status, info.p, 4, cudaMemcpyDeviceToHost, c.stream));
    check(cudaMemcpyAsync(a.values, values.p, size_t(a.dimension) * 8, cudaMemcpyDeviceToHost, c.stream));
    check(cudaMemcpyAsync(a.vectors, matrix.p, dd * 8, cudaMemcpyDeviceToHost, c.stream));
    check(cudaStreamSynchronize(c.stream));
    require(status == 0, "Eigensolver did not converge");
    std::reverse(a.values, a.values + a.dimension);
    for (int i = 0; i < a.dimension / 2; ++i)
        for (int j = 0; j < a.dimension; ++j)
            std::swap(a.vectors[size_t(i) * a.dimension + j],
                      a.vectors[size_t(a.dimension - 1 - i) * a.dimension + j]);
}
void runPreprocess(Context& c, const anom_compute_preprocess& a) {
    require(a.images && a.count && a.count <= INT_MAX, "Invalid image batch");
    elements(a.resize_h, a.resize_w);
    size_t n = elements(int(a.count), a.crop_h, a.crop_w, 3);
    require(
        a.crop_h <= a.resize_h && a.crop_w <= a.resize_w && (a.interpolation == 0 || a.interpolation == 1),
        "Unsupported resize or crop");
    for (int j = 0; j < 3; ++j)
        require(std::isfinite(a.mean[j]) && std::isfinite(a.stddev[j]) && a.stddev[j] > 0,
                "Invalid normalization");
    Argument out(c, a.output, n * 4, true);
    for (size_t i = 0; i < a.count; ++i) {
        auto im = a.images[i];
        elements(im.height, im.width);
        require(im.data && (im.channels == 1 || im.channels == 3 || im.channels == 4) &&
                    im.stride >= size_t(im.width) * im.channels && im.stride <= SIZE_MAX / size_t(im.height),
                "Invalid image");
        size_t bytes = size_t(im.width) * im.channels * im.height;
        Buffer<unsigned char> raw(bytes);
        check(cudaMemcpy2DAsync(raw.p, size_t(im.width) * im.channels, im.data, im.stride,
                                size_t(im.width) * im.channels, im.height, cudaMemcpyHostToDevice, c.stream));
        preprocess<<<blocks(size_t(a.crop_h) * a.crop_w * 3), 256, 0, c.stream>>>(
            raw.p, out.as<float>(), size_t(im.width) * im.channels, im.height, im.width, im.channels, a,
            int(i));
    }
    out.finish(n * 4);
}
void runMap(Context& c, const anom_compute_map& a) {
    size_t in = elements(a.n, a.height, a.width), out = elements(a.n, a.output_h, a.output_w);
    require(a.gaussian && a.kernel > 0 && (a.kernel % 2) == 1 && a.kernel <= 4095, "Invalid Gaussian kernel");
    Argument x(c, a.input, in * 4), y(c, a.output, out * 4, true);
    if (a.kernel == 1)
        resizeMaps<<<blocks(out), 256, 0, c.stream>>>(x.as<float>(), y.as<float>(), out, a.height, a.width,
                                                      a.output_h, a.output_w);
    else {
        Buffer<float> resized(out), tmp(out), weights(a.kernel);
        check(cudaMemcpyAsync(weights.p, a.gaussian, size_t(a.kernel) * 4, cudaMemcpyHostToDevice, c.stream));
        resizeMaps<<<blocks(out), 256, 0, c.stream>>>(x.as<float>(), resized.p, out, a.height, a.width,
                                                      a.output_h, a.output_w);
        gaussian<<<blocks(out), 256, 0, c.stream>>>(resized.p, tmp.p, weights.p, out, a.output_h, a.output_w,
                                                    a.kernel, false);
        gaussian<<<blocks(out), 256, 0, c.stream>>>(tmp.p, y.as<float>(), weights.p, out, a.output_h,
                                                    a.output_w, a.kernel, true);
    }
    y.finish(out * 4);
}
void runScores(Context& c, const anom_compute_scores& a) {
    size_t n = elements(a.n, a.locations, a.neighbors), out = elements(a.n, a.locations);
    require(a.maxima && a.argmax, "Missing score outputs");
    Argument d(c, a.distances, n * 4), p(c, a.scores, out * 4, true);
    Buffer<float> maximum(a.n);
    Buffer<int64_t> index(a.n);
    Buffer<int> invalid(1);
    check(cudaMemsetAsync(invalid.p, 0, 4, c.stream));
    scores<<<a.n, 256, 0, c.stream>>>(d.as<float>(), p.as<float>(), maximum.p, index.p, invalid.p,
                                      a.locations, a.neighbors, a.square_root != 0);
    int bad;
    check(cudaMemcpyAsync(&bad, invalid.p, 4, cudaMemcpyDeviceToHost, c.stream));
    check(cudaMemcpyAsync(a.maxima, maximum.p, size_t(a.n) * 4, cudaMemcpyDeviceToHost, c.stream));
    check(cudaMemcpyAsync(a.argmax, index.p, size_t(a.n) * 8, cudaMemcpyDeviceToHost, c.stream));
    p.finish(out * 4);
    check(cudaStreamSynchronize(c.stream));
    require(!bad, "Invalid neighbor distances");
}
template <class F>
int32_t guarded(F f) {
    lastError.clear();
    try {
        f();
        return 0;
    } catch (const std::invalid_argument& e) {
        lastError = e.what();
        return int32_t(ErrorCode::InvalidArgument);
    } catch (const std::bad_alloc& e) {
        lastError = e.what();
        return int32_t(ErrorCode::OutOfMemory);
    } catch (const std::exception& e) {
        lastError = e.what();
        return int32_t(ErrorCode::CudaFailure);
    } catch (...) {
        lastError = "Unknown CUDA plugin error";
        return int32_t(ErrorCode::CudaFailure);
    }
}
int32_t ANOM_PLUGIN_CALL create(int32_t device, void** out) {
    if (out) *out = nullptr;
    return guarded([&] {
        require(out && device >= 0, "Invalid device/output");
        int count = 0;
        check(cudaGetDeviceCount(&count));
        require(device < count, "Unavailable CUDA device");
        auto c = std::make_unique<Context>(device);
        c->init();
        *out = c.release();
    });
}
void ANOM_PLUGIN_CALL destroy(void* p) { delete static_cast<Context*>(p); }
int32_t ANOM_PLUGIN_CALL allocate(void* p, size_t n, void** out) {
    if (out) *out = nullptr;
    return guarded([&] {
        require(p && out && n, "Invalid allocation");
        auto& c = *static_cast<Context*>(p);
        std::lock_guard<std::mutex> lock(c.mutex);
        DeviceScope scope(c.device);
        check(cudaMalloc(out, n));
    });
}
void ANOM_PLUGIN_CALL release(void* p, void* ptr) {
    if (p && ptr) try {
            auto& c = *static_cast<Context*>(p);
            std::lock_guard<std::mutex> lock(c.mutex);
            DeviceScope scope(c.device);
            cudaFree(ptr);
        } catch (...) {
        }
}
int32_t ANOM_PLUGIN_CALL copy(void* p, anom_compute_buffer dst, anom_compute_buffer src) {
    return guarded([&] {
        require(p && dst.data && src.data && dst.bytes >= src.bytes, "Invalid copy");
        auto& c = *static_cast<Context*>(p);
        std::lock_guard<std::mutex> lock(c.mutex);
        DeviceScope scope(c.device);
        require((dst.device == -1 || dst.device == c.device) && (src.device == -1 || src.device == c.device),
                "Copy device mismatch");
        Completion done{c.stream};
        cudaMemcpyKind kind = src.device < 0
                                  ? (dst.device < 0 ? cudaMemcpyHostToHost : cudaMemcpyHostToDevice)
                                  : (dst.device < 0 ? cudaMemcpyDeviceToHost : cudaMemcpyDeviceToDevice);
        check(cudaMemcpyAsync(dst.data, src.data, src.bytes, kind, c.stream));
        check(cudaStreamSynchronize(c.stream));
    });
}
int32_t ANOM_PLUGIN_CALL run(void* p, int32_t op, const void* args, size_t size) {
    return guarded([&] {
        require(p && args, "Null operation");
        auto& c = *static_cast<Context*>(p);
        std::lock_guard<std::mutex> lock(c.mutex);
        DeviceScope scope(c.device);
        Completion done{c.stream};
#define OP(id, type, fn)                                                   \
    case id:                                                               \
        require(size == sizeof(type), "Invalid operation structure size"); \
        fn(c, *static_cast<const type*>(args));                            \
        break
        switch (op) {
            OP(ANOM_COMPUTE_AGGREGATE, anom_compute_aggregate, runAggregate);
            OP(ANOM_COMPUTE_PROJECT, anom_compute_project, runProject);
            OP(ANOM_COMPUTE_CORESET, anom_compute_coreset, runCoreset);
            OP(ANOM_COMPUTE_TRANSFORM, anom_compute_transform, runTransform);
            OP(ANOM_COMPUTE_MOMENTS, anom_compute_moments, runMoments);
            OP(ANOM_COMPUTE_EIGEN, anom_compute_eigen, runEigen);
            OP(ANOM_COMPUTE_PREPROCESS, anom_compute_preprocess, runPreprocess);
            OP(ANOM_COMPUTE_MAP, anom_compute_map, runMap);
            OP(ANOM_COMPUTE_SCORES, anom_compute_scores, runScores);
            default:
                throw std::invalid_argument("Unknown compute operation");
        }
#undef OP
        check(cudaGetLastError());
        check(cudaStreamSynchronize(c.stream));
    });
}
size_t ANOM_PLUGIN_CALL error(char* out, size_t size) {
    size_t n = lastError.size() + 1;
    if (out && size) {
        size_t take = std::min(size - 1, lastError.size());
        std::memcpy(out, lastError.data(), take);
        out[take] = 0;
    }
    return n;
}
}  // namespace
extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_compute_query_v1(uint32_t version,
                                                                             anom_compute_api_v1* api) {
    if (version != 1 || !api || api->struct_size < sizeof(*api)) return int32_t(ErrorCode::PluginAbiMismatch);
    *api = {sizeof(*api), 1, create, destroy, allocate, release, copy, run, error};
    return 0;
}
