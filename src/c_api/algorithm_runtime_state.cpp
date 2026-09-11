#define ANOM_ENGINE_ENABLE_LEGACY_SESSION_API 1
#include "anomEngine/common.h"
#include "c_api/algorithm_runtime_state.h"

#include <new>

namespace anom::c_api {

AlgorithmRuntimeState::~AlgorithmRuntimeState() {
    anom_session_destroy(session);
    anom_fitter_destroy(fitter);
}

bool attachRuntime(void*& internal, anom_session* session) noexcept {
    auto* state = static_cast<AlgorithmRuntimeState*>(internal);
    if (!state) {
        state = new (std::nothrow) AlgorithmRuntimeState;
        if (!state) {
            anom_session_destroy(session);
            return false;
        }
        internal = state;
    }
    state->session = session;
    return true;
}

bool attachRuntime(void*& internal, anom_fitter* fitter) noexcept {
    auto* state = static_cast<AlgorithmRuntimeState*>(internal);
    if (!state) {
        state = new (std::nothrow) AlgorithmRuntimeState;
        if (!state) {
            anom_fitter_destroy(fitter);
            return false;
        }
        internal = state;
    }
    state->fitter = fitter;
    return true;
}

}  // namespace anom::c_api
