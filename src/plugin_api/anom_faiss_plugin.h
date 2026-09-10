#ifndef ANOM_ENGINE_FAISS_PLUGIN_H
#define ANOM_ENGINE_FAISS_PLUGIN_H
#include "plugin_api/anom_plugin_common.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Optional GPU retrieval plugin. All buffers are host memory; search completes
 * transfers before returning. No Faiss/C++ objects cross this boundary. */
typedef struct anom_faiss_api_v1 {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t (ANOM_PLUGIN_CALL *probe)(int32_t device_id);
    int32_t (ANOM_PLUGIN_CALL *create)(const char* index_path_utf8, int32_t device_id,
        int32_t dimension, int64_t vector_count, int32_t neighbors, void** instance);
    void (ANOM_PLUGIN_CALL *destroy)(void* instance);
    int32_t (ANOM_PLUGIN_CALL *search)(void* instance, int64_t count,
        const float* queries, int32_t neighbors, float* distances, int64_t* labels);
    size_t (ANOM_PLUGIN_CALL *get_last_error)(void* instance, char* buffer, size_t size);
} anom_faiss_api_v1;
typedef int32_t (ANOM_PLUGIN_CALL *anom_faiss_query_v1_fn)(uint32_t version, anom_faiss_api_v1* api);
#ifdef __cplusplus
}
#endif
#endif
