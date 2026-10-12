#ifndef ANOM_DEVICE_PLUGIN_H
#define ANOM_DEVICE_PLUGIN_H
#include "plugin_api/anom_algorithm_plugin.h"
#include "plugin_api/anom_backend_plugin.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Explicit negotiated device ABI. view.data is CUDA only when device_id >= 0;
 * -1 means host. Producer synchronizes before returning. Ownership lasts until
 * release_batch; host retains the instance and library. Legacy v1 APIs remain
 * host-only. No stream handles or C++ objects cross this boundary. */
typedef struct anom_device_tensor_view_v1 {
    anom_plugin_tensor_view_v1 view;
    int32_t device_id;
} anom_device_tensor_view_v1;
typedef struct anom_device_batch_v1 {
    const anom_device_tensor_view_v1* tensors;
    size_t count;
    void* owner;
} anom_device_batch_v1;
typedef struct anom_backend_device_api_v1 {
    uint32_t struct_size, abi_version;
    int32_t(ANOM_PLUGIN_CALL* infer)(void*, const anom_device_tensor_view_v1*, size_t, anom_device_batch_v1*);
    void(ANOM_PLUGIN_CALL* release_batch)(anom_device_batch_v1*);
} anom_backend_device_api_v1;
typedef int32_t(ANOM_PLUGIN_CALL* anom_backend_query_device_v1_fn)(uint32_t, anom_backend_device_api_v1*);
typedef struct anom_algorithm_device_api_v1 {
    uint32_t struct_size, abi_version;
    int32_t(ANOM_PLUGIN_CALL* configure)(void*, int32_t, const char*);
    int32_t(ANOM_PLUGIN_CALL* predict)(void*, const anom_device_tensor_view_v1*, size_t, int32_t, int32_t,
                                       anom_plugin_prediction_batch_v1*);
} anom_algorithm_device_api_v1;
typedef int32_t(ANOM_PLUGIN_CALL* anom_algorithm_query_device_v1_fn)(uint32_t, anom_algorithm_device_api_v1*);
#ifdef __cplusplus
}
#endif
#endif
