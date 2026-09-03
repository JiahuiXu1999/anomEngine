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
    ANOM_STATUS_CANCELLED = 10,
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
typedef struct anom_model anom_model_t;
typedef struct anom_package_builder anom_package_builder_t;
typedef struct anom_fitter anom_fitter_t;
typedef struct anom_calibrator anom_calibrator_t;

typedef uint64_t anom_algorithm_capabilities_t;
#define ANOM_ALGORITHM_CAP_PACKAGE_IMPORT  (UINT64_C(1) << 0)
#define ANOM_ALGORITHM_CAP_ARTIFACT_FIT    (UINT64_C(1) << 1)
#define ANOM_ALGORITHM_CAP_INCREMENTAL_FIT (UINT64_C(1) << 2)
#define ANOM_ALGORITHM_CAP_CALIBRATION     (UINT64_C(1) << 3)
#define ANOM_ALGORITHM_CAP_EVALUATION      (UINT64_C(1) << 4)
#define ANOM_ALGORITHM_CAP_WEIGHT_TRAINING (UINT64_C(1) << 5)

typedef struct anom_algorithm_info {
    uint32_t struct_size;
    const char* algorithm_utf8;
    anom_algorithm_capabilities_t capabilities;
    int32_t available;
    uint32_t reserved[8];
} anom_algorithm_info_t;

typedef struct anom_session_options {
    uint32_t struct_size;
    const char* plugin_directory_utf8;
    int32_t warmup;
    uint32_t reserved[8];
} anom_session_options_t;

typedef int32_t anom_device_preference_t;
enum {
    ANOM_DEVICE_AUTO = 0,
    ANOM_DEVICE_CPU = 1,
    ANOM_DEVICE_GPU = 2
};

typedef int32_t anom_fallback_policy_t;
enum {
    ANOM_FALLBACK_NONE = 0,
    ANOM_FALLBACK_LOAD_ONLY = 1
};

typedef int32_t anom_precision_t;
enum {
    ANOM_PRECISION_AUTO = 0,
    ANOM_PRECISION_FP32 = 1,
    ANOM_PRECISION_FP16 = 2
};

/*
 * Session options with explicit execution-device selection. AUTO tries
 * TensorRT/CUDA, then ONNX Runtime/CPU. A non-empty
 * backend_utf8 constrains selection to "tensorrt" or "onnxruntime".
 */
typedef struct anom_session_options_v2 {
    uint32_t struct_size;
    const char* plugin_directory_utf8;
    int32_t warmup;
    anom_device_preference_t device;
    int32_t device_id; /* -1 selects the runtime default device. */
    anom_fallback_policy_t fallback;
    anom_precision_t precision;
    const char* backend_utf8;
    uint32_t reserved[8];
} anom_session_options_v2_t;

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

typedef struct anom_execution_info {
    uint32_t struct_size;
    anom_device_preference_t requested_device;
    const char* backend_utf8;
    const char* execution_provider_utf8;
    int32_t device_id;
    const char* device_name_utf8;
    anom_precision_t precision;
    int32_t fallback_occurred;
    const char* fallback_reason_utf8;
    uint32_t reserved[8];
} anom_execution_info_t;

typedef struct anom_model_validate_options {
    uint32_t struct_size;
    const char* plugin_directory_utf8;
    int32_t initialize_runtime;
    uint32_t reserved[8];
} anom_model_validate_options_t;

typedef struct anom_model_validation_report {
    uint32_t struct_size;
    int32_t package_valid;
    int32_t runtime_valid;
    uint32_t reserved[8];
} anom_model_validation_report_t;

/*
 * A package builder clones a manifest/package template, overlays explicitly
 * supplied artifacts, generates checksums, validates the result, and commits
 * it atomically to a new output directory.
 */
typedef struct anom_package_builder_options {
    uint32_t struct_size;
    const char* template_package_utf8;
    uint32_t reserved[8];
} anom_package_builder_options_t;

typedef struct anom_fitter_options {
    uint32_t struct_size;
    const char* template_package_utf8;
    const char* plugin_directory_utf8;

    float patchcore_coreset_sampling_ratio;
    size_t patchcore_projection_dimension;
    uint64_t patchcore_max_exact_distance_evaluations;
    uint32_t random_seed;

    double padim_covariance_regularization;
    const int32_t* padim_channel_indices;
    size_t padim_channel_index_count;

    uint32_t reserved[8];
} anom_fitter_options_t;

typedef int32_t anom_fit_stage_t;
enum {
    ANOM_FIT_STAGE_READY = 0,
    ANOM_FIT_STAGE_EXTRACTING = 1,
    ANOM_FIT_STAGE_FINALIZING = 2,
    ANOM_FIT_STAGE_COMPLETE = 3,
    ANOM_FIT_STAGE_CANCELLED = 4
};

typedef struct anom_fit_progress {
    uint32_t struct_size;
    anom_fit_stage_t stage;
    uint64_t processed_samples;
    uint64_t collected_items;
    int32_t cancellation_requested;
    uint32_t reserved[8];
} anom_fit_progress_t;

typedef struct anom_calibrator_options {
    uint32_t struct_size;
    const char* plugin_directory_utf8;
    float target_false_positive_rate;
    uint64_t max_pixel_samples;
    uint32_t random_seed;
    uint32_t reserved[8];
} anom_calibrator_options_t;

typedef struct anom_calibration_result {
    uint32_t struct_size;
    uint64_t sample_count;
    float image_min;
    float image_max;
    float image_threshold;
    float pixel_min;
    float pixel_max;
    float pixel_threshold;
    int32_t has_pixel_statistics;
    uint32_t reserved[8];
} anom_calibration_result_t;

/*
 * Model-info string pointers remain valid until the session or model handle
 * that produced them is destroyed.
 */

ANOM_ENGINE_API uint32_t anom_get_abi_version(void);
ANOM_ENGINE_API const char* anom_get_version_string(void);

ANOM_ENGINE_API size_t anom_algorithm_get_count(void);
ANOM_ENGINE_API anom_status_t anom_algorithm_get_info(
    size_t index,
    anom_algorithm_info_t* out_info);

ANOM_ENGINE_API anom_status_t anom_model_open(
    const char* model_package_utf8,
    anom_model_t** out_model);

ANOM_ENGINE_API anom_status_t anom_model_get_info(
    const anom_model_t* model,
    anom_model_info_t* out_info);

ANOM_ENGINE_API anom_status_t anom_model_validate(
    const char* model_package_utf8,
    const anom_model_validate_options_t* options,
    anom_model_validation_report_t* out_report);

ANOM_ENGINE_API void anom_model_destroy(anom_model_t* model);

ANOM_ENGINE_API anom_status_t anom_package_builder_create(
    const anom_package_builder_options_t* options,
    anom_package_builder_t** out_builder);

ANOM_ENGINE_API anom_status_t anom_package_builder_add_artifact(
    anom_package_builder_t* builder,
    const char* source_path_utf8,
    const char* package_relative_path_utf8);

ANOM_ENGINE_API anom_status_t anom_package_builder_commit(
    anom_package_builder_t* builder,
    const char* output_package_utf8);

ANOM_ENGINE_API anom_status_t anom_package_builder_apply_calibration(
    anom_package_builder_t* builder,
    const anom_calibration_result_t* calibration);

ANOM_ENGINE_API void anom_package_builder_destroy(
    anom_package_builder_t* builder);

ANOM_ENGINE_API anom_status_t anom_fitter_create(
    const anom_fitter_options_t* options,
    anom_fitter_t** out_fitter);

ANOM_ENGINE_API anom_status_t anom_fitter_add_batch(
    anom_fitter_t* fitter,
    const anom_image_t* images,
    size_t image_count);

ANOM_ENGINE_API anom_status_t anom_fitter_get_progress(
    const anom_fitter_t* fitter,
    anom_fit_progress_t* out_progress);

ANOM_ENGINE_API anom_status_t anom_fitter_save_checkpoint(
    const anom_fitter_t* fitter,
    const char* checkpoint_path_utf8);

ANOM_ENGINE_API anom_status_t anom_fitter_load_checkpoint(
    anom_fitter_t* fitter,
    const char* checkpoint_path_utf8);

ANOM_ENGINE_API anom_status_t anom_fitter_cancel(anom_fitter_t* fitter);

ANOM_ENGINE_API anom_status_t anom_fitter_finalize(
    anom_fitter_t* fitter,
    const char* output_package_utf8);

ANOM_ENGINE_API void anom_fitter_destroy(anom_fitter_t* fitter);

ANOM_ENGINE_API anom_status_t anom_calibrator_create(
    const anom_model_t* model,
    const anom_calibrator_options_t* options,
    anom_calibrator_t** out_calibrator);

/* Calibration batches are expected to contain representative normal images. */
ANOM_ENGINE_API anom_status_t anom_calibrator_add_batch(
    anom_calibrator_t* calibrator,
    const anom_image_t* images,
    size_t image_count);

ANOM_ENGINE_API anom_status_t anom_calibrator_compute(
    const anom_calibrator_t* calibrator,
    anom_calibration_result_t* out_result);

ANOM_ENGINE_API void anom_calibrator_destroy(anom_calibrator_t* calibrator);

ANOM_ENGINE_API anom_status_t anom_session_create(
    const char* model_package_utf8,
    const anom_session_options_t* options,
    anom_session_t** out_session);

ANOM_ENGINE_API anom_status_t anom_session_create_v2(
    const char* model_package_utf8,
    const anom_session_options_v2_t* options,
    anom_session_t** out_session);

ANOM_ENGINE_API void anom_session_destroy(anom_session_t* session);

ANOM_ENGINE_API anom_status_t anom_session_warmup(anom_session_t* session);

ANOM_ENGINE_API anom_status_t anom_session_get_model_info(
    const anom_session_t* session,
    anom_model_info_t* out_info);

ANOM_ENGINE_API anom_status_t anom_session_get_execution_info(
    const anom_session_t* session,
    anom_execution_info_t* out_info);

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
