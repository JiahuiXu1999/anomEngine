// Runtime test doubles for the private object-method implementations.
#include "c_api/algorithm_object_impl.h"

#include <string>

thread_local std::string anomLastError;

namespace anom::c_api {

anom_status_t ANOM_CALL directLoad(
    anom_direct_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL directRelease(anom_direct_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL directWarmup(anom_direct_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directGetModelInfo(
    const anom_direct_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directGetExecutionInfo(
    const anom_direct_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directPredict(
    anom_direct_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directPredictBatch(
    anom_direct_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadLoad(
    anom_efficientad_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL efficientadRelease(anom_efficientad_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL efficientadWarmup(anom_efficientad_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadGetModelInfo(
    const anom_efficientad_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadGetExecutionInfo(
    const anom_efficientad_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadPredict(
    anom_efficientad_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadPredictBatch(
    anom_efficientad_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdeLoad(
    anom_dfkde_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL dfkdeRelease(anom_dfkde_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL dfkdeWarmup(anom_dfkde_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdeGetModelInfo(
    const anom_dfkde_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdeGetExecutionInfo(
    const anom_dfkde_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdePredict(
    anom_dfkde_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdePredictBatch(
    anom_dfkde_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimLoad(
    anom_padim_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL padimRelease(anom_padim_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL padimWarmup(anom_padim_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimGetModelInfo(
    const anom_padim_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimGetExecutionInfo(
    const anom_padim_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimPredict(
    anom_padim_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimPredictBatch(
    anom_padim_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreLoad(
    anom_patchcore_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL patchcoreRelease(anom_patchcore_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL patchcoreWarmup(anom_patchcore_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreGetModelInfo(
    const anom_patchcore_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreGetExecutionInfo(
    const anom_patchcore_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcorePredict(
    anom_patchcore_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcorePredictBatch(
    anom_patchcore_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadeLoad(
    anom_spade_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL spadeRelease(anom_spade_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL spadeWarmup(anom_spade_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadeGetModelInfo(
    const anom_spade_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadeGetExecutionInfo(
    const anom_spade_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadePredict(
    anom_spade_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadePredictBatch(
    anom_spade_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloLoad(
    anom_yolo_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || self->internal || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

void ANOM_CALL yoloRelease(anom_yolo_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL yoloWarmup(anom_yolo_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloGetModelInfo(
    const anom_yolo_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloGetExecutionInfo(
    const anom_yolo_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloPredict(
    anom_yolo_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloPredictBatch(
    anom_yolo_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterCreate(
    anom_padim_fitter_t* self, const anom_padim_fitter_options_t* options) {
    if (!self || self->internal || !options ||
        options->struct_size != sizeof(*options)) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

anom_status_t ANOM_CALL padimFitterAddBatch(
    anom_padim_fitter_t* self, const anom_image_t* images, size_t imageCount) {
    (void)images;
    (void)imageCount;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterGetProgress(
    const anom_padim_fitter_t* self, anom_fit_progress_t* outProgress) {
    (void)outProgress;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterSaveCheckpoint(
    const anom_padim_fitter_t* self, const char* path) {
    (void)path;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterLoadCheckpoint(
    anom_padim_fitter_t* self, const char* path) {
    (void)path;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterCancel(anom_padim_fitter_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterFinalize(
    anom_padim_fitter_t* self, const char* outputPackage) {
    (void)outputPackage;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

void ANOM_CALL padimFitterRelease(anom_padim_fitter_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

anom_status_t ANOM_CALL patchcoreFitterCreate(
    anom_patchcore_fitter_t* self, const anom_patchcore_fitter_options_t* options) {
    if (!self || self->internal || !options ||
        options->struct_size != sizeof(*options)) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    self->internal = self;
    return ANOM_STATUS_OK;
}

anom_status_t ANOM_CALL patchcoreFitterAddBatch(
    anom_patchcore_fitter_t* self, const anom_image_t* images, size_t imageCount) {
    (void)images;
    (void)imageCount;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterGetProgress(
    const anom_patchcore_fitter_t* self, anom_fit_progress_t* outProgress) {
    (void)outProgress;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterSaveCheckpoint(
    const anom_patchcore_fitter_t* self, const char* path) {
    (void)path;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterLoadCheckpoint(
    anom_patchcore_fitter_t* self, const char* path) {
    (void)path;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterCancel(anom_patchcore_fitter_t* self) {
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterFinalize(
    anom_patchcore_fitter_t* self, const char* outputPackage) {
    (void)outputPackage;
    return self && self->internal == self ? ANOM_STATUS_OK
                                         : ANOM_STATUS_INVALID_ARGUMENT;
}

void ANOM_CALL patchcoreFitterRelease(anom_patchcore_fitter_t* self) {
    if (self && self->struct_size >= sizeof(*self)) self->internal = nullptr;
}

}  // namespace anom::c_api
