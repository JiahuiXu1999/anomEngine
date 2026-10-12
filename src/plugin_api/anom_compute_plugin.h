#ifndef ANOM_COMPUTE_PLUGIN_H
#define ANOM_COMPUTE_PLUGIN_H
#include "plugin_api/anom_plugin_common.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Separate optional ABI: v1 backend/algorithm host views never contain CUDA
 * pointers. All operations are synchronous, including device outputs. Buffers
 * belong to the caller, may not alias, and device buffers must use this device.
 * An instance serializes calls and restores the calling thread's CUDA device. */
typedef struct anom_compute_buffer {
    void* data;
    size_t bytes;
    int32_t device;
} anom_compute_buffer;
/* device = -1 for host, otherwise CUDA ordinal. */
typedef struct anom_compute_layer {
    anom_compute_buffer buffer;
    int32_t n, c, h, w;
} anom_compute_layer;
typedef struct anom_compute_aggregate {
    const anom_compute_layer* layers;
    size_t count;
    int32_t kernel, stride, padding, mean;
    anom_compute_buffer output; /* row-major [N*H*W, C], first pooled grid */
} anom_compute_aggregate;
typedef struct anom_compute_project {
    anom_compute_buffer input, matrix, output;
    int32_t rows, dimension, rank; /* float X[N,D] R[D,K] -> Y[N,K] */
} anom_compute_project;
typedef struct anom_compute_coreset {
    anom_compute_buffer input;
    int32_t rows, dimension, count, first;
    int64_t* selected; /* host indices; ties resolve to lowest index */
} anom_compute_coreset;
typedef struct anom_compute_transform {
    anom_compute_buffer input, output;
    int32_t rows, dimension, rank, residual;
    const double *mean, *basis, *scales; /* host, basis is [rank,dimension] */
} anom_compute_transform;
typedef struct anom_compute_moments {
    anom_compute_buffer input;
    int32_t rows, dimension;
    double *mean, *m2; /* host outputs, population centered sum of products */
} anom_compute_moments;
typedef struct anom_compute_eigen {
    int32_t dimension;
    const double* covariance;
    double *values, *vectors; /* host, descending values, eigenvectors as rows */
} anom_compute_eigen;
typedef struct anom_compute_image {
    const unsigned char* data;
    size_t stride;
    int32_t height, width, channels;
} anom_compute_image;
typedef struct anom_compute_preprocess {
    const anom_compute_image* images;
    size_t count;
    int32_t resize_h, resize_w, crop_h, crop_w, rgb, nchw, interpolation;
    float mean[3], stddev[3];
    anom_compute_buffer output;
} anom_compute_preprocess;
typedef struct anom_compute_map {
    anom_compute_buffer input, output;
    int32_t n, height, width, output_h, output_w;
    const float* gaussian;
    int32_t kernel; /* host coefficients; 1 = no blur */
} anom_compute_map;
typedef struct anom_compute_scores {
    anom_compute_buffer distances, scores;
    int32_t n, locations, neighbors, square_root;
    float* maxima;
    int64_t* argmax; /* host, one per image */
} anom_compute_scores;
enum {
    ANOM_COMPUTE_AGGREGATE = 1,
    ANOM_COMPUTE_PROJECT,
    ANOM_COMPUTE_CORESET,
    ANOM_COMPUTE_TRANSFORM,
    ANOM_COMPUTE_MOMENTS,
    ANOM_COMPUTE_EIGEN,
    ANOM_COMPUTE_PREPROCESS,
    ANOM_COMPUTE_MAP,
    ANOM_COMPUTE_SCORES
};
typedef struct anom_compute_api_v1 {
    uint32_t struct_size, abi_version;
    int32_t(ANOM_PLUGIN_CALL* create)(int32_t device, void** instance);
    void(ANOM_PLUGIN_CALL* destroy)(void* instance);
    int32_t(ANOM_PLUGIN_CALL* allocate)(void* instance, size_t bytes, void** data);
    void(ANOM_PLUGIN_CALL* release)(void* instance, void* data);
    int32_t(ANOM_PLUGIN_CALL* copy)(void* instance, anom_compute_buffer dst, anom_compute_buffer src);
    int32_t(ANOM_PLUGIN_CALL* run)(void* instance, int32_t operation, const void* args, size_t size);
    size_t(ANOM_PLUGIN_CALL* error)(char* buffer, size_t size);
} anom_compute_api_v1;
typedef int32_t(ANOM_PLUGIN_CALL* anom_compute_query_v1_fn)(uint32_t, anom_compute_api_v1*);
#ifdef __cplusplus
}
#endif
#endif
