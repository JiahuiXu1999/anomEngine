#include "anomEngine/direct.h"
#include "anomEngine/anomEngine.h"

int anom_direct_header_compile_test(void) {
    anom_direct_t object = {0};
    object.struct_size = sizeof(object);
    return anom_direct_init(&object);
}
