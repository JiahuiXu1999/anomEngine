#ifndef ANOM_ENGINE_PLUGIN_COMMON_H
#define ANOM_ENGINE_PLUGIN_COMMON_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  define ANOM_PLUGIN_EXPORT __declspec(dllexport)
#  define ANOM_PLUGIN_CALL __cdecl
#elif defined(__GNUC__) || defined(__clang__)
#  define ANOM_PLUGIN_EXPORT __attribute__((visibility("default")))
#  define ANOM_PLUGIN_CALL
#else
#  define ANOM_PLUGIN_EXPORT
#  define ANOM_PLUGIN_CALL
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define ANOM_PLUGIN_ABI_VERSION 1u

typedef struct anom_plugin_tensor_spec_v1 {
    const char* name_utf8;
    int32_t data_type;
    const int64_t* dimensions;
    size_t rank;
} anom_plugin_tensor_spec_v1;

typedef struct anom_plugin_signature_v1 {
    const anom_plugin_tensor_spec_v1* inputs;
    size_t input_count;
    const anom_plugin_tensor_spec_v1* outputs;
    size_t output_count;
} anom_plugin_signature_v1;

typedef struct anom_plugin_tensor_view_v1 {
    const char* name_utf8;
    int32_t data_type;
    const int64_t* dimensions;
    size_t rank;
    const void* data;
    size_t byte_size;
} anom_plugin_tensor_view_v1;

#ifdef __cplusplus
}
#endif

#endif
