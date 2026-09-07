#include "anomEngine/yolo.h"
#include "anomEngine/anomEngine.h"

int anom_yolo_header_compile_test(void) {
    anom_yolo_t object = {0};
    object.struct_size = sizeof(object);
    return anom_yolo_init(&object);
}
