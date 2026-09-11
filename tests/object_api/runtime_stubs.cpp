// Runtime test doubles for the private object-method implementations.
#define ANOM_ENGINE_ENABLE_LEGACY_SESSION_API 1
#include "c_api/algorithm_object_impl.h"
#include "c_api/algorithm_runtime_state.h"

#include <string>

thread_local std::string anomLastError;

struct anom_session {
    const void* owner;
    inline static size_t live = 0;
    explicit anom_session(const void* value) : owner(value) { ++live; }
    ~anom_session() { --live; }
};
struct anom_fitter {
    const void* owner;
    inline static size_t live = 0;
    explicit anom_fitter(const void* value) : owner(value) { ++live; }
    ~anom_fitter() { --live; }
};
extern "C" ANOM_ENGINE_API void ANOM_CALL anom_session_destroy(anom_session_t* session) {
    delete session;
}
extern "C" ANOM_ENGINE_API void ANOM_CALL anom_fitter_destroy(anom_fitter_t* fitter) {
    delete fitter;
}
extern "C" ANOM_ENGINE_API size_t ANOM_CALL test_live_sessions(void) {
    return anom_session::live;
}
extern "C" ANOM_ENGINE_API size_t ANOM_CALL test_live_fitters(void) {
    return anom_fitter::live;
}

namespace anom::c_api {
template <typename Object>
bool hasSession(const Object* self) {
    return self && self->struct_size >= sizeof(*self) && self->internal &&
           runtimeState(self)->session && runtimeState(self)->session->owner == self;
}
template <typename Object>
bool hasFitter(const Object* self) {
    return self && self->struct_size >= sizeof(*self) && self->internal &&
           runtimeState(self)->fitter && runtimeState(self)->fitter->owner == self;
}

anom_status_t ANOM_CALL directLoad(
    anom_direct_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL directRelease(anom_direct_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL directWarmup(anom_direct_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directGetModelInfo(
    const anom_direct_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directGetExecutionInfo(
    const anom_direct_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directPredict(
    anom_direct_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL directPredictBatch(
    anom_direct_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadLoad(
    anom_efficientad_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL efficientadRelease(anom_efficientad_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL efficientadWarmup(anom_efficientad_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadGetModelInfo(
    const anom_efficientad_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadGetExecutionInfo(
    const anom_efficientad_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadPredict(
    anom_efficientad_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL efficientadPredictBatch(
    anom_efficientad_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdeLoad(
    anom_dfkde_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL dfkdeRelease(anom_dfkde_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL dfkdeWarmup(anom_dfkde_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdeGetModelInfo(
    const anom_dfkde_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdeGetExecutionInfo(
    const anom_dfkde_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdePredict(
    anom_dfkde_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL dfkdePredictBatch(
    anom_dfkde_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimLoad(
    anom_padim_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL padimRelease(anom_padim_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL padimWarmup(anom_padim_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimGetModelInfo(
    const anom_padim_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimGetExecutionInfo(
    const anom_padim_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimPredict(
    anom_padim_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimPredictBatch(
    anom_padim_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreLoad(
    anom_patchcore_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL patchcoreRelease(anom_patchcore_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL patchcoreWarmup(anom_patchcore_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreGetModelInfo(
    const anom_patchcore_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreGetExecutionInfo(
    const anom_patchcore_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcorePredict(
    anom_patchcore_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcorePredictBatch(
    anom_patchcore_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadeLoad(
    anom_spade_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL spadeRelease(anom_spade_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL spadeWarmup(anom_spade_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadeGetModelInfo(
    const anom_spade_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadeGetExecutionInfo(
    const anom_spade_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadePredict(
    anom_spade_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL spadePredictBatch(
    anom_spade_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloLoad(
    anom_yolo_t* self, const char* modelPackage,
    const anom_algorithm_options_t* options) {
    if (!self || hasSession(self) || !modelPackage ||
        std::string(modelPackage) != "model" || !options || options->warmup != 7) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_session(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

void ANOM_CALL yoloRelease(anom_yolo_t* self) {
    if (self && self->struct_size >= sizeof(*self)) {
        delete runtimeState(self);
        self->internal = nullptr;
    }
}

anom_status_t ANOM_CALL yoloWarmup(anom_yolo_t* self) {
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloGetModelInfo(
    const anom_yolo_t* self, anom_model_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloGetExecutionInfo(
    const anom_yolo_t* self, anom_execution_info_t* outInfo) {
    (void)outInfo;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloPredict(
    anom_yolo_t* self, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    (void)image;
    (void)outPrediction;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL yoloPredictBatch(
    anom_yolo_t* self, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    (void)images;
    (void)imageCount;
    (void)outPredictions;
    return hasSession(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterCreate(
    anom_padim_t* self, const anom_padim_fitter_options_t* options) {
    if (!self || hasFitter(self) || !options ||
        options->struct_size != sizeof(*options)) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_fitter(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

anom_status_t ANOM_CALL padimFitterAddBatch(
    anom_padim_t* self, const anom_image_t* images, size_t imageCount) {
    (void)images;
    (void)imageCount;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterGetProgress(
    const anom_padim_t* self, anom_fit_progress_t* outProgress) {
    (void)outProgress;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterSaveCheckpoint(
    const anom_padim_t* self, const char* path) {
    (void)path;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterLoadCheckpoint(
    anom_padim_t* self, const char* path) {
    (void)path;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterCancel(anom_padim_t* self) {
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL padimFitterFinalize(
    anom_padim_t* self, const char* outputPackage) {
    (void)outputPackage;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterCreate(
    anom_patchcore_t* self, const anom_patchcore_fitter_options_t* options) {
    if (!self || hasFitter(self) || !options ||
        options->struct_size != sizeof(*options)) {
        return ANOM_STATUS_INVALID_ARGUMENT;
    }
    return attachRuntime(self->internal, new anom_fitter(self))
               ? ANOM_STATUS_OK : ANOM_STATUS_OUT_OF_MEMORY;
}

anom_status_t ANOM_CALL patchcoreFitterAddBatch(
    anom_patchcore_t* self, const anom_image_t* images, size_t imageCount) {
    (void)images;
    (void)imageCount;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterGetProgress(
    const anom_patchcore_t* self, anom_fit_progress_t* outProgress) {
    (void)outProgress;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterSaveCheckpoint(
    const anom_patchcore_t* self, const char* path) {
    (void)path;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterLoadCheckpoint(
    anom_patchcore_t* self, const char* path) {
    (void)path;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterCancel(anom_patchcore_t* self) {
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}

anom_status_t ANOM_CALL patchcoreFitterFinalize(
    anom_patchcore_t* self, const char* outputPackage) {
    (void)outputPackage;
    return hasFitter(self) ? ANOM_STATUS_OK : ANOM_STATUS_INVALID_ARGUMENT;
}


}  // namespace anom::c_api
