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
/* PaDiM operations preserve per-spatial-location statistics. */
typedef struct anom_compute_select_features {
    const anom_compute_layer* layers;
    size_t count;
    const int32_t* channels; /* host indices in concatenated channel order */
    int32_t dimension, height, width;
    anom_compute_buffer output; /* float32 [N,H,W,D] */
} anom_compute_select_features;
typedef struct anom_compute_padim_distance {
    anom_compute_buffer patches, mean, precision, output;
    int32_t n, locations, dimension; /* float32, precision [L,D,D] */
} anom_compute_padim_distance;
typedef struct anom_compute_padim_accumulate {
    anom_compute_buffer patches, sums, moments;
    int32_t n, locations, dimension, location_begin, location_count;
    /* patches float32 [N,L,D]; sums/moments float64 for this location chunk.
     * Existing sums/moments are updated in place, preserving population sums. */
} anom_compute_padim_accumulate;
typedef struct anom_compute_padim_finalize {
    anom_compute_buffer sums, moments, mean, precision;
    int32_t locations, dimension;
    uint64_t samples;
    double regularization;
    /* sums/moments float64; mean/precision float32, buffers for this chunk. */
} anom_compute_padim_finalize;

typedef struct anom_compute_channel_l2 {
    anom_compute_layer input, reference;
    int32_t difference; /* 0: norm(input), 1: norm(input-reference), over channels */
    anom_compute_buffer output; /* float32 [N,1,H,W] */
} anom_compute_channel_l2;
typedef struct anom_compute_normalized_blend {
    anom_compute_buffer hard, soft, output; /* float32 [N,H,W] */
    int32_t n, height, width;
    float hard_weight, soft_weight; /* normalize each image/branch before weighted blending */
} anom_compute_normalized_blend;
typedef struct anom_compute_dfkde {
    anom_compute_buffer patches, mean, basis, gallery, output;
    int32_t rows, dimension, rank, gallery_count;
    float sigma;
    uint64_t workspace_bytes; /* bounds temporary KDE storage, excluding inputs/output */
    /* float32: patches[Q,D], mean[D], basis[D,K], gallery[G,K], output[Q]. */
} anom_compute_dfkde;
typedef struct anom_compute_capabilities {
    uint64_t required_operations; /* bit (1 << operation); success only if all are supported */
} anom_compute_capabilities;

enum {
    ANOM_COMPUTE_AGGREGATE = 1,
    ANOM_COMPUTE_PROJECT,
    ANOM_COMPUTE_CORESET,
    ANOM_COMPUTE_TRANSFORM,
    ANOM_COMPUTE_MOMENTS,
    ANOM_COMPUTE_EIGEN,
    ANOM_COMPUTE_PREPROCESS,
    ANOM_COMPUTE_MAP,
    ANOM_COMPUTE_SCORES,
    ANOM_COMPUTE_SELECT_FEATURES,
    ANOM_COMPUTE_PADIM_DISTANCE,
    ANOM_COMPUTE_PADIM_ACCUMULATE,
    ANOM_COMPUTE_PADIM_FINALIZE,
    ANOM_COMPUTE_PADIM_PROBE, /* uint32_t value 1; old plugins reject this operation */
    ANOM_COMPUTE_CHANNEL_L2,
    ANOM_COMPUTE_NORMALIZED_BLEND,
    ANOM_COMPUTE_DFKDE,
    ANOM_COMPUTE_CAPABILITIES
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
