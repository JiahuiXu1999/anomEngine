#ifndef ANOM_ENGINE_ALGORITHM_PLUGIN_H
#define ANOM_ENGINE_ALGORITHM_PLUGIN_H

#include "plugin_api/anom_plugin_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anom_plugin_raw_prediction_v1 {
    float score;
    const float* anomaly_map;
    int32_t map_width;
    int32_t map_height;
    int32_t map_stride_elements;
} anom_plugin_raw_prediction_v1;

typedef struct anom_plugin_prediction_batch_v1 {
    const anom_plugin_raw_prediction_v1* predictions;
    size_t count;
    void* owner;
} anom_plugin_prediction_batch_v1;

typedef struct anom_algorithm_plugin_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* algorithm_name_utf8;

    int32_t (ANOM_PLUGIN_CALL *create)(const char* model_package_utf8, void** out_instance);
    void (ANOM_PLUGIN_CALL *destroy)(void* instance);
    int32_t (ANOM_PLUGIN_CALL *validate_signature)(
        void* instance, const anom_plugin_signature_v1* signature);
    int32_t (ANOM_PLUGIN_CALL *predict)(
        void* instance,
        const anom_plugin_tensor_view_v1* outputs,
        size_t output_count,
        int32_t model_input_width,
        int32_t model_input_height,
        anom_plugin_prediction_batch_v1* out_batch);
    void (ANOM_PLUGIN_CALL *release_batch)(anom_plugin_prediction_batch_v1* batch);
    size_t (ANOM_PLUGIN_CALL *get_last_error)(void* instance, char* buffer, size_t buffer_size);
} anom_algorithm_plugin_api_v1;

typedef int32_t (ANOM_PLUGIN_CALL *anom_algorithm_plugin_query_v1_fn)(
    uint32_t host_abi_version,
    anom_algorithm_plugin_api_v1* out_api);

#ifdef __cplusplus
}
#endif

#endif
