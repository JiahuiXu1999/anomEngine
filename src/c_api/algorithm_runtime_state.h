#pragma once

struct anom_session;
struct anom_fitter;

namespace anom::c_api {

// Allocated lazily by load/create. The two runtimes have independent state;
// the public object's release operation owns both.
struct AlgorithmRuntimeState {
    anom_session* session{nullptr};
    anom_fitter* fitter{nullptr};

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

}  // namespace anom::c_api
