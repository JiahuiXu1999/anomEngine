#include "anomEngine/anomEngine.h"

int anom_c_header_compile_test(void) {
    anom_session_options_t options = {0};
    anom_image_t image = {0};
    anom_prediction_t prediction = {0};
    options.struct_size = (uint32_t)sizeof(options);
    image.struct_size = (uint32_t)sizeof(image);
    prediction.struct_size = (uint32_t)sizeof(prediction);
    return (int)(options.struct_size + image.struct_size + prediction.struct_size);
}
