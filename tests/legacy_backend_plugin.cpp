// Deliberately exports only the original v1 API. Used to ensure a new host
// never interprets an old CPU-only plugin as a CUDA-capable plugin.
#include "plugin_api/anom_backend_plugin.h"

#include <cstring>

namespace {
int instance;
const int64_t dimensions[] = {-1, 3, 2, 2};
const anom_plugin_tensor_spec_v1 input = {"input", 0, dimensions, 4};
const anom_plugin_tensor_spec_v1 output = {"output", 0, dimensions, 4};

int32_t ANOM_PLUGIN_CALL create(const anom_backend_config_v1*, void** result) {
    *result = &instance;
    return 0;
}
void ANOM_PLUGIN_CALL destroy(void*) {}
int32_t ANOM_PLUGIN_CALL signature(void*, anom_plugin_signature_v1* result) {
    *result = {&input, 1, &output, 1};
    return 0;
}
int32_t ANOM_PLUGIN_CALL batchSize(void*) { return 1; }
int32_t ANOM_PLUGIN_CALL infer(void*, const anom_plugin_tensor_view_v1*, size_t,
                               anom_plugin_tensor_batch_v1*) { return 1; }
void ANOM_PLUGIN_CALL release(anom_plugin_tensor_batch_v1*) {}
size_t ANOM_PLUGIN_CALL lastError(void*, char* buffer, size_t size) {
    if (buffer && size) buffer[0] = '\0';
    return 1;
}
}

extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_backend_plugin_query_v1(
    uint32_t version, anom_backend_plugin_api_v1* result) {
    if (version != 1 || !result || result->struct_size < sizeof(*result)) return -1;
    *result = {sizeof(*result), 1, "onnxruntime", &create, &destroy, &signature,
               &batchSize, &infer, &release, &lastError};
    return 0;
}
