#ifndef ANOM_ENGINE_ANOM_ENGINE_H
#define ANOM_ENGINE_ANOM_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(ANOM_ENGINE_BUILD)
#    define ANOM_ENGINE_API __declspec(dllexport)
#  else
#    define ANOM_ENGINE_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__) || defined(__clang__)
#  define ANOM_ENGINE_API __attribute__((visibility("default")))
#else
#  define ANOM_ENGINE_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ANOM_ENGINE_ABI_VERSION 1u

typedef int32_t anom_status_t;
enum {
    ANOM_STATUS_OK = 0,
    ANOM_STATUS_INVALID_ARGUMENT = 1,
    ANOM_STATUS_INVALID_IMAGE = 2,
    ANOM_STATUS_IO_ERROR = 3,
    ANOM_STATUS_INVALID_MODEL_PACKAGE = 4,
    ANOM_STATUS_UNSUPPORTED = 5,
    ANOM_STATUS_PLUGIN_NOT_FOUND = 6,
    ANOM_STATUS_PLUGIN_ABI_MISMATCH = 7,
    ANOM_STATUS_INFERENCE_FAILED = 8,
    ANOM_STATUS_OUT_OF_MEMORY = 9,
    ANOM_STATUS_INTERNAL_ERROR = 100
};

typedef int32_t anom_pixel_format_t;
enum {
    ANOM_PIXEL_FORMAT_GRAY8 = 1,
    ANOM_PIXEL_FORMAT_BGR8 = 2,
    ANOM_PIXEL_FORMAT_RGB8 = 3,
    ANOM_PIXEL_FORMAT_BGRA8 = 4,
    ANOM_PIXEL_FORMAT_RGBA8 = 5
};

typedef struct anom_session anom_session_t;

typedef struct anom_session_options {
    uint32_t struct_size;
    const char* plugin_directory_utf8;
    int32_t warmup;
    uint32_t reserved[8];
} anom_session_options_t;

typedef struct anom_image {
    uint32_t struct_size;
    const uint8_t* data;
    int32_t width;
    int32_t height;
    int32_t stride_bytes;
    anom_pixel_format_t pixel_format;
    uint32_t reserved[4];
} anom_image_t;

typedef struct anom_region {
    int32_t x;
    int32_t y;
    int32_t width;
    int32_t height;
    int32_t max_x;
    int32_t max_y;
    double area;
    float mean_score;
    float max_score;
} anom_region_t;

/*
 * All pointed-to result memory belongs to anomEngine.dll and stays valid until
 * anom_prediction_release is called. Zero-initialize this structure and then
 * set struct_size before prediction.
 */
typedef struct anom_prediction {
    uint32_t struct_size;
    float raw_score;
    float score;
    int32_t is_anomalous;

    const float* raw_anomaly_map;
    const float* anomaly_map;
    const uint8_t* mask;
    int32_t map_width;
    int32_t map_height;
    int32_t map_stride_elements;
    int32_t mask_stride_bytes;

    const anom_region_t* regions;
    size_t region_count;
    float anomaly_area_ratio;
    float mean_score;
    float max_score;
    int32_t has_map;

    const char* model_id_utf8;
    const char* model_version_utf8;

    double preprocess_ms;
    double backend_ms;
    double adapter_ms;
    double postprocess_ms;
    double total_ms;

    void* internal;
    uint32_t reserved[8];
} anom_prediction_t;

typedef struct anom_model_info {
    uint32_t struct_size;
    const char* model_id_utf8;
    const char* model_version_utf8;
    const char* algorithm_utf8;
    const char* backend_utf8;
    const char* execution_provider_utf8;
    uint32_t reserved[8];
} anom_model_info_t;

/* Model-info string pointers remain valid until the session is destroyed. */

ANOM_ENGINE_API uint32_t anom_get_abi_version(void);
ANOM_ENGINE_API const char* anom_get_version_string(void);

ANOM_ENGINE_API anom_status_t anom_session_create(
    const char* model_package_utf8,
    const anom_session_options_t* options,
    anom_session_t** out_session);

ANOM_ENGINE_API void anom_session_destroy(anom_session_t* session);

ANOM_ENGINE_API anom_status_t anom_session_warmup(anom_session_t* session);

ANOM_ENGINE_API anom_status_t anom_session_get_model_info(
    const anom_session_t* session,
    anom_model_info_t* out_info);

ANOM_ENGINE_API anom_status_t anom_session_predict(
    anom_session_t* session,
    const anom_image_t* image,
    anom_prediction_t* out_prediction);

ANOM_ENGINE_API anom_status_t anom_session_predict_batch(
    anom_session_t* session,
    const anom_image_t* images,
    size_t image_count,
    anom_prediction_t* out_predictions);

ANOM_ENGINE_API void anom_prediction_release(anom_prediction_t* prediction);

/* Returns the required byte count including the trailing NUL. */
ANOM_ENGINE_API size_t anom_get_last_error(char* buffer, size_t buffer_size);

#ifdef __cplusplus
}
#endif

#endif
