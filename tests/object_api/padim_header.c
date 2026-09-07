#include "anomEngine/padim.h"
#include "anomEngine/anomEngine.h"

int anom_padim_header_compile_test(void) {
    anom_padim_t object = {0};
    object.struct_size = sizeof(object);
    return anom_padim_init(&object);
}
