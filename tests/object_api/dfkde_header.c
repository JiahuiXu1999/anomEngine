#include "anomEngine/dfkde.h"
#include "anomEngine/anomEngine.h"

int anom_dfkde_header_compile_test(void) {
    anom_dfkde_t object = {0};
    object.struct_size = sizeof(object);
    return anom_dfkde_init(&object);
}
