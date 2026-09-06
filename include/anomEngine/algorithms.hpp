#pragma once

#include "anomEngine/anomEngine.h"

#include <cstddef>
#include <utility>

namespace anom {

/*
 * Thin, move-only C++ facades over the stable C ABI. Each algorithm is a
 * distinct type so algorithm-specific operations can be added without growing
 * a common session interface.
 */
#define ANOM_DEFINE_CPP_ALGORITHM(type_name, c_name)                              \
    struct type_name final {                                                     \
        type_name() noexcept { native.struct_size = sizeof(native); }            \
        ~type_name() { anom_##c_name##_release(&native); }                       \
        type_name(const type_name&) = delete;                                    \
        type_name& operator=(const type_name&) = delete;                         \
        type_name(type_name&& other) noexcept : native(other.native) {           \
            other.native = {};                                                   \
            other.native.struct_size = sizeof(other.native);                     \
        }                                                                        \
        type_name& operator=(type_name&& other) noexcept {                       \
            if (this != &other) {                                                \
                anom_##c_name##_release(&native);                                \
                native = other.native;                                           \
                other.native = {};                                               \
                other.native.struct_size = sizeof(other.native);                 \
            }                                                                    \
            return *this;                                                        \
        }                                                                        \
        anom_status_t load(                                                      \
            const char* modelPackageUtf8,                                        \
            const anom_algorithm_options_t* options = nullptr) {                 \
            return anom_##c_name##_load(modelPackageUtf8, options, &native);     \
        }                                                                        \
        void release() noexcept { anom_##c_name##_release(&native); }            \
        anom_status_t warmup() { return anom_##c_name##_warmup(&native); }       \
        anom_status_t modelInfo(anom_model_info_t& output) const {               \
            return anom_##c_name##_get_model_info(&native, &output);             \
        }                                                                        \
        anom_status_t executionInfo(anom_execution_info_t& output) const {       \
            return anom_##c_name##_get_execution_info(&native, &output);         \
        }                                                                        \
        anom_status_t predict(                                                   \
            const anom_image_t& image, anom_prediction_t& output) {              \
            return anom_##c_name##_predict(&native, &image, &output);            \
        }                                                                        \
        anom_status_t predictBatch(                                              \
            const anom_image_t* images, std::size_t imageCount,                  \
            anom_prediction_t* outputs) {                                        \
            return anom_##c_name##_predict_batch(                               \
                &native, images, imageCount, outputs);                           \
        }                                                                        \
        [[nodiscard]] bool loaded() const noexcept { return native.internal != nullptr; }\
        anom_##c_name##_t native{};                                              \
    }

ANOM_DEFINE_CPP_ALGORITHM(Direct, direct);
ANOM_DEFINE_CPP_ALGORITHM(EfficientAD, efficientad);
ANOM_DEFINE_CPP_ALGORITHM(DFKDE, dfkde);
ANOM_DEFINE_CPP_ALGORITHM(PaDiM, padim);
ANOM_DEFINE_CPP_ALGORITHM(PatchCore, patchcore);
ANOM_DEFINE_CPP_ALGORITHM(SPADE, spade);
ANOM_DEFINE_CPP_ALGORITHM(Yolo, yolo);

#undef ANOM_DEFINE_CPP_ALGORITHM

#define ANOM_DEFINE_CPP_FITTER(type_name, c_name)                                \
    struct type_name final {                                                     \
        type_name() noexcept { native.struct_size = sizeof(native); }            \
        ~type_name() { anom_##c_name##_fitter_release(&native); }                \
        type_name(const type_name&) = delete;                                    \
        type_name& operator=(const type_name&) = delete;                         \
        type_name(type_name&& other) noexcept : native(other.native) {           \
            other.native = {};                                                   \
            other.native.struct_size = sizeof(other.native);                     \
        }                                                                        \
        type_name& operator=(type_name&& other) noexcept {                       \
            if (this != &other) {                                                \
                anom_##c_name##_fitter_release(&native);                         \
                native = other.native;                                           \
                other.native = {};                                               \
                other.native.struct_size = sizeof(other.native);                 \
            }                                                                    \
            return *this;                                                        \
        }                                                                        \
        anom_status_t create(const anom_##c_name##_fitter_options_t& options) {  \
            return anom_##c_name##_fitter_create(&options, &native);             \
        }                                                                        \
        anom_status_t addBatch(                                                  \
            const anom_image_t* images, std::size_t imageCount) {                \
            return anom_##c_name##_fitter_add_batch(&native, images, imageCount);\
        }                                                                        \
        anom_status_t progress(anom_fit_progress_t& output) const {              \
            return anom_##c_name##_fitter_get_progress(&native, &output);        \
        }                                                                        \
        anom_status_t saveCheckpoint(const char* pathUtf8) const {               \
            return anom_##c_name##_fitter_save_checkpoint(&native, pathUtf8);    \
        }                                                                        \
        anom_status_t loadCheckpoint(const char* pathUtf8) {                     \
            return anom_##c_name##_fitter_load_checkpoint(&native, pathUtf8);    \
        }                                                                        \
        anom_status_t cancel() { return anom_##c_name##_fitter_cancel(&native); }\
        anom_status_t finalize(const char* outputPackageUtf8) {                  \
            return anom_##c_name##_fitter_finalize(&native, outputPackageUtf8);  \
        }                                                                        \
        void release() noexcept { anom_##c_name##_fitter_release(&native); }     \
        [[nodiscard]] bool initialized() const noexcept {                        \
            return native.internal != nullptr;                                  \
        }                                                                        \
        anom_##c_name##_fitter_t native{};                                       \
    }

ANOM_DEFINE_CPP_FITTER(PatchCoreFitter, patchcore);
ANOM_DEFINE_CPP_FITTER(PaDiMFitter, padim);

#undef ANOM_DEFINE_CPP_FITTER

}  // namespace anom
