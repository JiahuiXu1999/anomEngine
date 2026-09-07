#include "anomEngine/patchcore.h"
#include "anomEngine/anomEngine.h"

int anom_patchcore_header_compile_test(void) {
    anom_patchcore_t object = {0};
    object.struct_size = sizeof(object);
    return anom_patchcore_init(&object);
}
