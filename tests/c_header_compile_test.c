#include "anomEngine/anomEngine.h"

int anom_c_header_compile_test(void) {
    anom_algorithm_options_t options = {0};
    anom_direct_t direct = {0};
    anom_efficientad_t efficientad = {0};
    anom_dfkde_t dfkde = {0};
    anom_padim_t padim = {0};
    anom_patchcore_t patchcore = {0};
    anom_spade_t spade = {0};
    anom_yolo_t yolo = {0};
    anom_execution_info_t execution = {0};
    anom_image_t image = {0};
    anom_prediction_t prediction = {0};
    anom_algorithm_info_t algorithm = {0};
    anom_model_validation_report_t validation = {0};
    anom_package_builder_options_t builder = {0};
    anom_patchcore_fitter_t patchcore_fitter = {0};
    anom_patchcore_fitter_options_t patchcore_fitter_options = {0};
    anom_padim_fitter_t padim_fitter = {0};
    anom_padim_fitter_options_t padim_fitter_options = {0};
    anom_fit_progress_t progress = {0};
    anom_calibrator_options_t calibrator = {0};
    anom_calibration_result_t calibration = {0};
    options.struct_size = (uint32_t)sizeof(options);
    direct.struct_size = (uint32_t)sizeof(direct);
    efficientad.struct_size = (uint32_t)sizeof(efficientad);
    dfkde.struct_size = (uint32_t)sizeof(dfkde);
    padim.struct_size = (uint32_t)sizeof(padim);
    patchcore.struct_size = (uint32_t)sizeof(patchcore);
    spade.struct_size = (uint32_t)sizeof(spade);
    yolo.struct_size = (uint32_t)sizeof(yolo);
    execution.struct_size = (uint32_t)sizeof(execution);
    image.struct_size = (uint32_t)sizeof(image);
    prediction.struct_size = (uint32_t)sizeof(prediction);
    algorithm.struct_size = (uint32_t)sizeof(algorithm);
    validation.struct_size = (uint32_t)sizeof(validation);
    builder.struct_size = (uint32_t)sizeof(builder);
    patchcore_fitter.struct_size = (uint32_t)sizeof(patchcore_fitter);
    patchcore_fitter_options.struct_size = (uint32_t)sizeof(patchcore_fitter_options);
    padim_fitter.struct_size = (uint32_t)sizeof(padim_fitter);
    padim_fitter_options.struct_size = (uint32_t)sizeof(padim_fitter_options);
    progress.struct_size = (uint32_t)sizeof(progress);
    calibrator.struct_size = (uint32_t)sizeof(calibrator);
    calibration.struct_size = (uint32_t)sizeof(calibration);
    return (int)(options.struct_size + direct.struct_size + efficientad.struct_size +
                 dfkde.struct_size + padim.struct_size + patchcore.struct_size +
                 spade.struct_size + yolo.struct_size + execution.struct_size +
                 image.struct_size + prediction.struct_size + algorithm.struct_size +
                 validation.struct_size + builder.struct_size +
                 patchcore_fitter.struct_size + patchcore_fitter_options.struct_size +
                 padim_fitter.struct_size + padim_fitter_options.struct_size +
                 progress.struct_size + calibrator.struct_size + calibration.struct_size);
}
