#ifndef ANOM_ENGINE_DFKDE_H
#define ANOM_ENGINE_DFKDE_H

#include "anomEngine/common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anom_dfkde anom_dfkde_t;

/*
 * Zero-initialize, set struct_size = sizeof(object), then call anom_dfkde_init.
 * Check init's status before calling any member. Pass this object as self.
 * Do not copy an initialized object or modify internal/function-pointer fields.
 * release frees runtime state and preserves the method table for reuse.
 * Keep anomEngine loaded for the entire lifetime of the object and its results.
 */
struct anom_dfkde {
    uint32_t struct_size;
    void* internal;
    uint32_t reserved[8];
    uint32_t abi_version; /* Set by init to ANOM_ENGINE_ABI_VERSION. */

    anom_status_t (ANOM_CALL *load)(
        anom_dfkde_t* self, const char* model_package_utf8, const anom_algorithm_options_t* options);
    void (ANOM_CALL *release)(
        anom_dfkde_t* self);
    anom_status_t (ANOM_CALL *warmup)(
        anom_dfkde_t* self);
    anom_status_t (ANOM_CALL *get_model_info)(
        const anom_dfkde_t* self, anom_model_info_t* out_info);
    anom_status_t (ANOM_CALL *get_execution_info)(
        const anom_dfkde_t* self, anom_execution_info_t* out_info);
    anom_status_t (ANOM_CALL *predict)(
        anom_dfkde_t* self, const anom_image_t* image, anom_prediction_t* out_prediction);
    anom_status_t (ANOM_CALL *predict_batch)(
        anom_dfkde_t* self, const anom_image_t* images, size_t image_count, anom_prediction_t* out_predictions);
};

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_dfkde_init(anom_dfkde_t* self);

#ifdef __cplusplus
}
#endif

#endif
