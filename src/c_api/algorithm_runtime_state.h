#pragma once
#include "anomEngine/patchcore.h"

struct anom_session;
struct anom_fitter;

namespace anom::c_api {

// Allocated lazily by load/create. The two runtimes have independent state;
// the public object's release operation owns both.
struct AlgorithmRuntimeState {
    anom_session* session{nullptr};
    anom_fitter* fitter{nullptr};
    int patchcoreVariant{-1};

    AlgorithmRuntimeState() = default;
    AlgorithmRuntimeState(const AlgorithmRuntimeState&) = delete;
    AlgorithmRuntimeState& operator=(const AlgorithmRuntimeState&) = delete;
    ~AlgorithmRuntimeState();
};

// Take ownership even on allocation failure. Callers reject a duplicate runtime
// before constructing the candidate; a failed attachment preserves existing state.
bool attachRuntime(void*& internal, anom_session* session) noexcept;
bool attachRuntime(void*& internal, anom_fitter* fitter) noexcept;

template <typename Object>
AlgorithmRuntimeState* runtimeState(Object* object) noexcept {
    return static_cast<AlgorithmRuntimeState*>(object->internal);
}

template <typename Object>
const AlgorithmRuntimeState* runtimeState(const Object* object) noexcept {
    return static_cast<const AlgorithmRuntimeState*>(object->internal);
}

template <typename Object> bool validVariant(const Object*) noexcept { return true; }
inline bool validVariant(const anom_patchcore_t* object) noexcept {
    if (!object || object->variant < ANOM_PATCHCORE_ORIGINAL || object->variant > ANOM_PATCHCORE_SUBSPACE_AD) return false;
    const auto* state = runtimeState(object);
    return !state || state->patchcoreVariant < 0 || state->patchcoreVariant == static_cast<int>(object->variant);
}
template <typename Object> void rememberVariant(Object*) noexcept {}
inline void rememberVariant(anom_patchcore_t* object) noexcept {
    runtimeState(object)->patchcoreVariant = static_cast<int>(object->variant);
}

}  // namespace anom::c_api
