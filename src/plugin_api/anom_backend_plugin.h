#ifndef ANOM_ENGINE_BACKEND_PLUGIN_H
#define ANOM_ENGINE_BACKEND_PLUGIN_H

#include "plugin_api/anom_plugin_common.h"

#ifdef __cplusplus
extern "C" {
#endif

typedef struct anom_backend_config_v1 {
    uint32_t struct_size;
    const char* onnx_path_utf8;
    const char* engine_path_utf8;
    int32_t load_policy;
    int32_t fp16;
    int32_t max_batch_size;
    uint64_t workspace_bytes;
    const char* input_name_utf8;
    int32_t input_layout;
    int32_t input_height;
    int32_t input_width;
    int32_t reserved0;
    int32_t ort_graph_optimization;
    int32_t ort_execution_mode;
    int32_t reserved1;
    int32_t device_id;
    int32_t intra_op_threads;
    int32_t inter_op_threads;
    int32_t enable_memory_pattern;
    int32_t enable_cpu_memory_arena;
    int32_t enable_profiling;
    const char* profile_file_prefix_utf8;
} anom_backend_config_v1;

typedef struct anom_plugin_tensor_batch_v1 {
    const anom_plugin_tensor_view_v1* tensors;
    size_t count;
    void* owner;
} anom_plugin_tensor_batch_v1;

typedef struct anom_backend_plugin_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    const char* backend_name_utf8;

    int32_t (ANOM_PLUGIN_CALL *create)(const anom_backend_config_v1* config, void** out_instance);
    void (ANOM_PLUGIN_CALL *destroy)(void* instance);
    int32_t (ANOM_PLUGIN_CALL *get_signature)(
        void* instance, anom_plugin_signature_v1* out_signature);
    int32_t (ANOM_PLUGIN_CALL *get_max_batch_size)(void* instance);
    int32_t (ANOM_PLUGIN_CALL *infer)(
        void* instance,
        const anom_plugin_tensor_view_v1* inputs,
        size_t input_count,
        anom_plugin_tensor_batch_v1* out_batch);
    void (ANOM_PLUGIN_CALL *release_batch)(anom_plugin_tensor_batch_v1* batch);
    size_t (ANOM_PLUGIN_CALL *get_last_error)(void* instance, char* buffer, size_t buffer_size);
} anom_backend_plugin_api_v1;

typedef int32_t (ANOM_PLUGIN_CALL *anom_backend_plugin_query_v1_fn)(
    uint32_t host_abi_version,
    anom_backend_plugin_api_v1* out_api);

#ifdef __cplusplus
}
#endif

#endif
