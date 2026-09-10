#include "plugin_api/anom_backend_plugin.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <vector>

namespace {
int liveInstances = 0;
int liveBatches = 0;
int releasedBatches = 0;
int invalidRelease = 0;
const void* lastData = nullptr;
const int64_t dimensions[] = {1};
const anom_plugin_tensor_spec_v1 input = {"input", 0, dimensions, 1};
const anom_plugin_tensor_spec_v1 output = {"output", 0, dimensions, 1};
struct Instance { int batches = 0; };
struct Batch {
    Instance* instance;
    int64_t count = 1024 * 1024;
    std::vector<float> data = std::vector<float>(static_cast<size_t>(count), 3.25F);
    std::array<anom_plugin_tensor_view_v1, 2> views{};
};
int32_t ANOM_PLUGIN_CALL create(const anom_backend_config_v1*, void** result) {
    *result = new Instance;
    ++liveInstances;
    return 0;
}
void ANOM_PLUGIN_CALL destroy(void* value) {
    auto* instance = static_cast<Instance*>(value);
    if (instance->batches) ++invalidRelease;
    delete instance;
    --liveInstances;
}
int32_t ANOM_PLUGIN_CALL signature(void*, anom_plugin_signature_v1* result) {
    *result = {&input, 1, &output, 1};
    return 0;
}
int32_t ANOM_PLUGIN_CALL batchSize(void*) { return 1; }
int32_t ANOM_PLUGIN_CALL infer(void* value, const anom_plugin_tensor_view_v1* inputs,
                              size_t count, anom_plugin_tensor_batch_v1* result) {
    auto* instance = static_cast<Instance*>(value);
    auto* batch = new Batch;
    batch->instance = instance;
    ++instance->batches;
    ++liveBatches;
    lastData = batch->data.data();
    batch->views[0] = {"output", 0, &batch->count, 1, lastData, batch->data.size() * sizeof(float)};
    batch->views[1] = {"broken", 0, &batch->count, 1, nullptr, sizeof(float)};
    size_t outputCount = 1;
    if (count && std::strcmp(inputs[0].name_utf8, "invalid") == 0) outputCount = 2;
    if (count && std::strcmp(inputs[0].name_utf8, "empty") == 0) outputCount = 0;
    *result = {batch->views.data(), outputCount, batch};
    return 0;
}
void ANOM_PLUGIN_CALL release(anom_plugin_tensor_batch_v1* view) {
    auto* batch = static_cast<Batch*>(view->owner);
    if (!liveInstances) ++invalidRelease;
    else --batch->instance->batches;
    delete batch;
    --liveBatches;
    ++releasedBatches;
    *view = {};
}
size_t ANOM_PLUGIN_CALL lastError(void*, char* buffer, size_t size) {
    if (buffer && size) buffer[0] = '\0';
    return 1;
}
}

extern "C" ANOM_PLUGIN_EXPORT int ANOM_PLUGIN_CALL memory_stat(int field) {
    switch (field) {
        case 0: return liveInstances;
        case 1: return liveBatches;
        case 2: return releasedBatches;
        default: return invalidRelease;
    }
}
extern "C" ANOM_PLUGIN_EXPORT const void* ANOM_PLUGIN_CALL memory_last_data() { return lastData; }
extern "C" ANOM_PLUGIN_EXPORT int32_t ANOM_PLUGIN_CALL anom_backend_plugin_query_v1(
    uint32_t version, anom_backend_plugin_api_v1* result) {
    if (version != 1 || !result || result->struct_size < sizeof(*result)) return -1;
    *result = {sizeof(*result), 1, "onnxruntime", &create, &destroy, &signature,
              &batchSize, &infer, &release, &lastError};
    return 0;
}
