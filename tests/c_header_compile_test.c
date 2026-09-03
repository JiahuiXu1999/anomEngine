#include "anomEngine/anomEngine.h"

int anom_c_header_compile_test(void) {
    anom_session_options_t options = {0};
    anom_session_options_v2_t options_v2 = {0};
    anom_execution_info_t execution = {0};
    anom_image_t image = {0};
    anom_prediction_t prediction = {0};
    anom_algorithm_info_t algorithm = {0};
    anom_model_validation_report_t validation = {0};
    anom_package_builder_options_t builder = {0};
    anom_fitter_options_t fitter = {0};
    anom_fit_progress_t progress = {0};
    anom_calibrator_options_t calibrator = {0};
    anom_calibration_result_t calibration = {0};
    options.struct_size = (uint32_t)sizeof(options);
    options_v2.struct_size = (uint32_t)sizeof(options_v2);
    execution.struct_size = (uint32_t)sizeof(execution);
    image.struct_size = (uint32_t)sizeof(image);
    prediction.struct_size = (uint32_t)sizeof(prediction);
    algorithm.struct_size = (uint32_t)sizeof(algorithm);
    validation.struct_size = (uint32_t)sizeof(validation);
    builder.struct_size = (uint32_t)sizeof(builder);
    fitter.struct_size = (uint32_t)sizeof(fitter);
    progress.struct_size = (uint32_t)sizeof(progress);
    calibrator.struct_size = (uint32_t)sizeof(calibrator);
    calibration.struct_size = (uint32_t)sizeof(calibration);
    return (int)(options.struct_size + options_v2.struct_size + execution.struct_size +
                 image.struct_size + prediction.struct_size +
                 algorithm.struct_size + validation.struct_size + builder.struct_size +
                 fitter.struct_size + progress.struct_size + calibrator.struct_size +
                 calibration.struct_size);
}
