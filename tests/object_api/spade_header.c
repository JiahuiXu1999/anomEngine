#include "anomEngine/spade.h"
#include "anomEngine/anomEngine.h"

int anom_spade_header_compile_test(void) {
    anom_spade_t object = {0};
    object.struct_size = sizeof(object);
    return anom_spade_init(&object);
}
