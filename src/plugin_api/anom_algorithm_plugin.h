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

/* Optional extension; keeps the original algorithm plugin ABI unchanged.
 * Provider: 0 = no Faiss stage (output only), 1 = CPU, 2 = CUDA.
 * Configure only during load. Prediction never changes provider. */
typedef struct anom_search_config_v1 {
    uint32_t struct_size;
    int32_t provider;
    int32_t device_id;
    int32_t allow_cpu_fallback;
    const char* plugin_directory_utf8;
} anom_search_config_v1;
typedef struct anom_search_info_v1 {
    uint32_t struct_size;
    int32_t provider;
    int32_t device_id;
    int32_t fallback_occurred;
    const char* fallback_reason_utf8;
} anom_search_info_v1;
typedef struct anom_algorithm_execution_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (ANOM_PLUGIN_CALL *configure_search)(void* instance,
        const anom_search_config_v1* config, anom_search_info_v1* info);
} anom_algorithm_execution_api_v1;
typedef int32_t (ANOM_PLUGIN_CALL *anom_algorithm_query_execution_v1_fn)(
    uint32_t version, anom_algorithm_execution_api_v1* api);

#ifdef __cplusplus
}
#endif

#endif
