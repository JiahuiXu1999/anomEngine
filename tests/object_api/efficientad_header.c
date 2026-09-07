#include "anomEngine/efficientad.h"
#include "anomEngine/anomEngine.h"

int anom_efficientad_header_compile_test(void) {
    anom_efficientad_t object = {0};
    object.struct_size = sizeof(object);
    return anom_efficientad_init(&object);
}
