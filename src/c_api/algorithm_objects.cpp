#include "c_api/algorithm_object_impl.h"

#include <string>

extern thread_local std::string anomLastError;

namespace {

anom_status_t initError(const char* message) noexcept {
    try {
        anomLastError = message;
    } catch (...) {
        return ANOM_STATUS_OUT_OF_MEMORY;
    }
    return ANOM_STATUS_INVALID_ARGUMENT;
}

template <typename Object>
anom_status_t validateInit(Object* self) noexcept {
    anomLastError.clear();
    if (!self || self->struct_size < sizeof(Object)) {
        return initError("Object struct_size is incompatible with ABI v4");
    }
    if (self->abi_version != 0 && self->abi_version != ANOM_ENGINE_ABI_VERSION) {
        return initError("Object ABI version is incompatible");
    }
    if (self->internal) {
        return initError("Release the object before initializing it again");
    }
    return ANOM_STATUS_OK;
}

}  // namespace

extern "C" {

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_direct_init(anom_direct_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_direct_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::directLoad;
    initialized.release = anom::c_api::directRelease;
    initialized.warmup = anom::c_api::directWarmup;
    initialized.get_model_info = anom::c_api::directGetModelInfo;
    initialized.get_execution_info = anom::c_api::directGetExecutionInfo;
    initialized.predict = anom::c_api::directPredict;
    initialized.predict_batch = anom::c_api::directPredictBatch;
    *self = initialized;
    return ANOM_STATUS_OK;
}

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_efficientad_init(anom_efficientad_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_efficientad_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::efficientadLoad;
    initialized.release = anom::c_api::efficientadRelease;
    initialized.warmup = anom::c_api::efficientadWarmup;
    initialized.get_model_info = anom::c_api::efficientadGetModelInfo;
    initialized.get_execution_info = anom::c_api::efficientadGetExecutionInfo;
    initialized.predict = anom::c_api::efficientadPredict;
    initialized.predict_batch = anom::c_api::efficientadPredictBatch;
    *self = initialized;
    return ANOM_STATUS_OK;
}

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_dfkde_init(anom_dfkde_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_dfkde_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::dfkdeLoad;
    initialized.release = anom::c_api::dfkdeRelease;
    initialized.warmup = anom::c_api::dfkdeWarmup;
    initialized.get_model_info = anom::c_api::dfkdeGetModelInfo;
    initialized.get_execution_info = anom::c_api::dfkdeGetExecutionInfo;
    initialized.predict = anom::c_api::dfkdePredict;
    initialized.predict_batch = anom::c_api::dfkdePredictBatch;
    *self = initialized;
    return ANOM_STATUS_OK;
}

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_padim_init(anom_padim_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_padim_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::padimLoad;
    initialized.release = anom::c_api::padimRelease;
    initialized.warmup = anom::c_api::padimWarmup;
    initialized.get_model_info = anom::c_api::padimGetModelInfo;
    initialized.get_execution_info = anom::c_api::padimGetExecutionInfo;
    initialized.predict = anom::c_api::padimPredict;
    initialized.predict_batch = anom::c_api::padimPredictBatch;
    initialized.create = anom::c_api::padimFitterCreate;
    initialized.add_batch = anom::c_api::padimFitterAddBatch;
    initialized.get_progress = anom::c_api::padimFitterGetProgress;
    initialized.save_checkpoint = anom::c_api::padimFitterSaveCheckpoint;
    initialized.load_checkpoint = anom::c_api::padimFitterLoadCheckpoint;
    initialized.cancel = anom::c_api::padimFitterCancel;
    initialized.finalize = anom::c_api::padimFitterFinalize;
    *self = initialized;
    return ANOM_STATUS_OK;
}

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_patchcore_init(anom_patchcore_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_patchcore_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::patchcoreLoad;
    initialized.release = anom::c_api::patchcoreRelease;
    initialized.warmup = anom::c_api::patchcoreWarmup;
    initialized.get_model_info = anom::c_api::patchcoreGetModelInfo;
    initialized.get_execution_info = anom::c_api::patchcoreGetExecutionInfo;
    initialized.predict = anom::c_api::patchcorePredict;
    initialized.predict_batch = anom::c_api::patchcorePredictBatch;
    initialized.create = anom::c_api::patchcoreFitterCreate;
    initialized.add_batch = anom::c_api::patchcoreFitterAddBatch;
    initialized.get_progress = anom::c_api::patchcoreFitterGetProgress;
    initialized.save_checkpoint = anom::c_api::patchcoreFitterSaveCheckpoint;
    initialized.load_checkpoint = anom::c_api::patchcoreFitterLoadCheckpoint;
    initialized.cancel = anom::c_api::patchcoreFitterCancel;
    initialized.finalize = anom::c_api::patchcoreFitterFinalize;
    *self = initialized;
    return ANOM_STATUS_OK;
}

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_spade_init(anom_spade_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_spade_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::spadeLoad;
    initialized.release = anom::c_api::spadeRelease;
    initialized.warmup = anom::c_api::spadeWarmup;
    initialized.get_model_info = anom::c_api::spadeGetModelInfo;
    initialized.get_execution_info = anom::c_api::spadeGetExecutionInfo;
    initialized.predict = anom::c_api::spadePredict;
    initialized.predict_batch = anom::c_api::spadePredictBatch;
    *self = initialized;
    return ANOM_STATUS_OK;
}

ANOM_ENGINE_API anom_status_t ANOM_CALL anom_yolo_init(anom_yolo_t* self) {
    const auto status = validateInit(self);
    if (status != ANOM_STATUS_OK) return status;

    anom_yolo_t initialized{};
    initialized.struct_size = self->struct_size;
    initialized.abi_version = ANOM_ENGINE_ABI_VERSION;
    initialized.load = anom::c_api::yoloLoad;
    initialized.release = anom::c_api::yoloRelease;
    initialized.warmup = anom::c_api::yoloWarmup;
    initialized.get_model_info = anom::c_api::yoloGetModelInfo;
    initialized.get_execution_info = anom::c_api::yoloGetExecutionInfo;
    initialized.predict = anom::c_api::yoloPredict;
    initialized.predict_batch = anom::c_api::yoloPredictBatch;
    *self = initialized;
    return ANOM_STATUS_OK;
}

}  // extern "C"
