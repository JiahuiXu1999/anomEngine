#ifndef ANOM_ENGINE_PATCHCORE_H
#define ANOM_ENGINE_PATCHCORE_H

#include "anomEngine/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anom_patchcore anom_patchcore_t;

/*
 * Zero-initialize, set struct_size = sizeof(object), then call anom_patchcore_init.
 * Check init's status before calling any member. Pass this object as self.
 * Do not copy an initialized object or modify internal/function-pointer fields.
 * release frees runtime state and preserves the method table for reuse.
 * Keep anomEngine loaded for the entire lifetime of the object and its results.
 */
struct anom_patchcore {
    uint32_t struct_size;
    void* internal;
    uint32_t reserved[8];
    uint32_t abi_version; /* Set by init to ANOM_ENGINE_ABI_VERSION. */

    anom_status_t (ANOM_CALL *load)(
        anom_patchcore_t* self, const char* model_package_utf8, const anom_algorithm_options_t* options);
    void (ANOM_CALL *release)(
        anom_patchcore_t* self);
    anom_status_t (ANOM_CALL *warmup)(
        anom_patchcore_t* self);
    anom_status_t (ANOM_CALL *get_model_info)(
        const anom_patchcore_t* self, anom_model_info_t* out_info);
    anom_status_t (ANOM_CALL *get_execution_info)(
        const anom_patchcore_t* self, anom_execution_info_t* out_info);
    anom_status_t (ANOM_CALL *predict)(
        anom_patchcore_t* self, const anom_image_t* image, anom_prediction_t* out_prediction);
    anom_status_t (ANOM_CALL *predict_batch)(
        anom_patchcore_t* self, const anom_image_t* images, size_t image_count, anom_prediction_t* out_predictions);
};

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_patchcore_init(anom_patchcore_t* self);

typedef struct anom_patchcore_fitter_options {
    uint32_t struct_size;
    const char* template_package_utf8;
    const char* plugin_directory_utf8;
    float coreset_sampling_ratio;
    size_t projection_dimension;
    uint64_t max_exact_distance_evaluations;
    uint32_t random_seed;
    uint32_t reserved[8];
} anom_patchcore_fitter_options_t;

typedef struct anom_patchcore_fitter anom_patchcore_fitter_t;

/*
 * Zero-initialize, set struct_size = sizeof(object), then call anom_patchcore_fitter_init.
 * Check init's status before calling any member. Pass this object as self.
 * Do not copy an initialized object or modify internal/function-pointer fields.
 * release frees runtime state and preserves the method table for reuse.
 * Keep anomEngine loaded for the entire lifetime of the object and its results.
 */
struct anom_patchcore_fitter {
    uint32_t struct_size;
    void* internal;
    uint32_t reserved[8];
    uint32_t abi_version; /* Set by init to ANOM_ENGINE_ABI_VERSION. */

    anom_status_t (ANOM_CALL *create)(
        anom_patchcore_fitter_t* self, const anom_patchcore_fitter_options_t* options);
    anom_status_t (ANOM_CALL *add_batch)(
        anom_patchcore_fitter_t* self, const anom_image_t* images, size_t image_count);
    anom_status_t (ANOM_CALL *get_progress)(
        const anom_patchcore_fitter_t* self, anom_fit_progress_t* out_progress);
    anom_status_t (ANOM_CALL *save_checkpoint)(
        const anom_patchcore_fitter_t* self, const char* checkpoint_path_utf8);
    anom_status_t (ANOM_CALL *load_checkpoint)(
        anom_patchcore_fitter_t* self, const char* checkpoint_path_utf8);
    anom_status_t (ANOM_CALL *cancel)(
        anom_patchcore_fitter_t* self);
    anom_status_t (ANOM_CALL *finalize)(
        anom_patchcore_fitter_t* self, const char* output_package_utf8);
    void (ANOM_CALL *release)(
        anom_patchcore_fitter_t* self);
};

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_patchcore_fitter_init(anom_patchcore_fitter_t* self);

#ifdef __cplusplus
}
#endif

#endif
