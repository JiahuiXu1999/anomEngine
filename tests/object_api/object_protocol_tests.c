#include "anomEngine/anomEngine.h"

#include <stdio.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "Check failed at line %d: %s\n", __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int test_direct(void) {
    anom_direct_t object = {0};
    anom_direct_t other = {0};
    anom_direct_t saved;
    struct extended_object {
        anom_direct_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_direct_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_direct_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_direct_init((anom_direct_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_direct_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_direct_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_direct_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_direct_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_direct_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_direct_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_direct_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_efficientad(void) {
    anom_efficientad_t object = {0};
    anom_efficientad_t other = {0};
    anom_efficientad_t saved;
    struct extended_object {
        anom_efficientad_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_efficientad_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_efficientad_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_efficientad_init((anom_efficientad_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_efficientad_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_efficientad_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_efficientad_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_efficientad_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_efficientad_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_efficientad_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_efficientad_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_dfkde(void) {
    anom_dfkde_t object = {0};
    anom_dfkde_t other = {0};
    anom_dfkde_t saved;
    struct extended_object {
        anom_dfkde_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_dfkde_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_dfkde_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_dfkde_init((anom_dfkde_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_dfkde_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_dfkde_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_dfkde_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_dfkde_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_dfkde_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_dfkde_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_dfkde_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_padim(void) {
    anom_padim_t object = {0};
    anom_padim_t other = {0};
    anom_padim_t saved;
    struct extended_object {
        anom_padim_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_padim_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_padim_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_padim_init((anom_padim_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_padim_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_padim_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_padim_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_padim_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_padim_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_padim_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_padim_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_padim_fitter(void) {
    anom_padim_fitter_t object = {0};
    anom_padim_fitter_t other = {0};
    anom_padim_fitter_t saved;
    struct extended_object {
        anom_padim_fitter_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_padim_fitter_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_padim_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_padim_fitter_init((anom_padim_fitter_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_padim_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_padim_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_padim_fitter_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.create != NULL);
    CHECK(object.add_batch != NULL);
    CHECK(object.get_progress != NULL);
    CHECK(object.save_checkpoint != NULL);
    CHECK(object.load_checkpoint != NULL);
    CHECK(object.cancel != NULL);
    CHECK(object.finalize != NULL);
    CHECK(object.release != NULL);
    CHECK(anom_padim_fitter_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_padim_fitter_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_padim_fitter_init(&other) == ANOM_STATUS_OK);
    {
        anom_padim_fitter_options_t options = {0};
        anom_fit_progress_t progress = {0};
        anom_image_t image = {0};
        options.struct_size = sizeof(options);
        CHECK(object.create(&object, NULL) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.create(&object, &options) == ANOM_STATUS_OK);
        CHECK(other.create(&other, &options) == ANOM_STATUS_OK);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_padim_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.add_batch(&object, &image, 1) == ANOM_STATUS_OK);
        CHECK(object.get_progress(&object, &progress) == ANOM_STATUS_OK);
        CHECK(object.save_checkpoint(&object, "checkpoint") == ANOM_STATUS_OK);
        CHECK(object.load_checkpoint(&object, "checkpoint") == ANOM_STATUS_OK);
        CHECK(object.finalize(&object, "output") == ANOM_STATUS_OK);
        CHECK(object.cancel(&object) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.get_progress(&other, &progress) == ANOM_STATUS_OK);
        CHECK(object.create(&object, &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_patchcore(void) {
    anom_patchcore_t object = {0};
    anom_patchcore_t other = {0};
    anom_patchcore_t saved;
    struct extended_object {
        anom_patchcore_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_patchcore_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_patchcore_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_patchcore_init((anom_patchcore_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_patchcore_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_patchcore_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_patchcore_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_patchcore_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_patchcore_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_patchcore_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_patchcore_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_patchcore_fitter(void) {
    anom_patchcore_fitter_t object = {0};
    anom_patchcore_fitter_t other = {0};
    anom_patchcore_fitter_t saved;
    struct extended_object {
        anom_patchcore_fitter_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_patchcore_fitter_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_patchcore_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_patchcore_fitter_init((anom_patchcore_fitter_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_patchcore_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_patchcore_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_patchcore_fitter_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.create != NULL);
    CHECK(object.add_batch != NULL);
    CHECK(object.get_progress != NULL);
    CHECK(object.save_checkpoint != NULL);
    CHECK(object.load_checkpoint != NULL);
    CHECK(object.cancel != NULL);
    CHECK(object.finalize != NULL);
    CHECK(object.release != NULL);
    CHECK(anom_patchcore_fitter_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_patchcore_fitter_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_patchcore_fitter_init(&other) == ANOM_STATUS_OK);
    {
        anom_patchcore_fitter_options_t options = {0};
        anom_fit_progress_t progress = {0};
        anom_image_t image = {0};
        options.struct_size = sizeof(options);
        CHECK(object.create(&object, NULL) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.create(&object, &options) == ANOM_STATUS_OK);
        CHECK(other.create(&other, &options) == ANOM_STATUS_OK);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_patchcore_fitter_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.add_batch(&object, &image, 1) == ANOM_STATUS_OK);
        CHECK(object.get_progress(&object, &progress) == ANOM_STATUS_OK);
        CHECK(object.save_checkpoint(&object, "checkpoint") == ANOM_STATUS_OK);
        CHECK(object.load_checkpoint(&object, "checkpoint") == ANOM_STATUS_OK);
        CHECK(object.finalize(&object, "output") == ANOM_STATUS_OK);
        CHECK(object.cancel(&object) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.get_progress(&other, &progress) == ANOM_STATUS_OK);
        CHECK(object.create(&object, &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_spade(void) {
    anom_spade_t object = {0};
    anom_spade_t other = {0};
    anom_spade_t saved;
    struct extended_object {
        anom_spade_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_spade_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_spade_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_spade_init((anom_spade_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_spade_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_spade_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_spade_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_spade_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_spade_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_spade_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_spade_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

static int test_yolo(void) {
    anom_yolo_t object = {0};
    anom_yolo_t other = {0};
    anom_yolo_t saved;
    struct extended_object {
        anom_yolo_t object;
        unsigned char tail[32];
    } extended;
    struct undersized_object {
        uint32_t struct_size;
        unsigned char canary[32];
        void* alignment;
    } undersized;
    unsigned char before[sizeof(undersized)];
    int state = 0;
    CHECK(anom_yolo_init(NULL) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(anom_yolo_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);

    memset(&undersized, 0xa5, sizeof(undersized));
    undersized.struct_size = sizeof(uint32_t);
    memcpy(before, &undersized, sizeof(before));
    CHECK(anom_yolo_init((anom_yolo_t*)&undersized) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(before, &undersized, sizeof(before)) == 0);

    object.struct_size = sizeof(object);
    object.abi_version = ANOM_ENGINE_ABI_VERSION + 1;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_yolo_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);

    object.abi_version = 0;
    object.internal = &state;
    memcpy(&saved, &object, sizeof(saved));
    CHECK(anom_yolo_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
    CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
    object.internal = NULL;

    CHECK(anom_yolo_init(&object) == ANOM_STATUS_OK);
    CHECK(object.struct_size == sizeof(object));
    CHECK(object.abi_version == ANOM_ENGINE_ABI_VERSION);
    CHECK(object.internal == NULL);
    CHECK(object.load != NULL);
    CHECK(object.release != NULL);
    CHECK(object.warmup != NULL);
    CHECK(object.get_model_info != NULL);
    CHECK(object.get_execution_info != NULL);
    CHECK(object.predict != NULL);
    CHECK(object.predict_batch != NULL);
    CHECK(anom_yolo_init(&object) == ANOM_STATUS_OK);

    memset(&extended, 0, sizeof(extended));
    memset(extended.tail, 0xa5, sizeof(extended.tail));
    extended.object.struct_size = sizeof(extended);
    CHECK(anom_yolo_init(&extended.object) == ANOM_STATUS_OK);
    CHECK(extended.object.struct_size == sizeof(extended));
    for (size_t i = 0; i < sizeof(extended.tail); ++i)
        CHECK(extended.tail[i] == 0xa5);

    other.struct_size = sizeof(other);
    CHECK(anom_yolo_init(&other) == ANOM_STATUS_OK);
    {
        anom_algorithm_options_t options = {0};
        anom_image_t image = {0};
        anom_prediction_t prediction = {0};
        anom_model_info_t model_info = {0};
        anom_execution_info_t execution_info = {0};
        options.struct_size = sizeof(options);
        options.warmup = 7;
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.load(&object, "bad path", &options) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(object.internal == NULL && object.load != NULL);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
        CHECK(other.load(&other, "model", &options) == ANOM_STATUS_OK);
        CHECK(object.internal != other.internal);
        memcpy(&saved, &object, sizeof(saved));
        CHECK(anom_yolo_init(&object) == ANOM_STATUS_INVALID_ARGUMENT);
        CHECK(memcmp(&saved, &object, sizeof(object)) == 0);
        CHECK(object.warmup(&object) == ANOM_STATUS_OK);
        CHECK(object.get_model_info(&object, &model_info) == ANOM_STATUS_OK);
        CHECK(object.get_execution_info(&object, &execution_info) == ANOM_STATUS_OK);
        CHECK(object.predict(&object, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.predict_batch(&object, &image, 1, &prediction) == ANOM_STATUS_OK);
        object.release(&object);
        CHECK(object.internal == NULL);
        CHECK(other.predict(&other, &image, &prediction) == ANOM_STATUS_OK);
        CHECK(object.load(&object, "model", &options) == ANOM_STATUS_OK);
    }
    object.release(&object);
    object.release(&object);
    other.release(&other);
    CHECK(object.internal == NULL && other.internal == NULL);
    CHECK(object.release != NULL && object.abi_version == ANOM_ENGINE_ABI_VERSION);
    return 0;
}

int ANOM_CALL main(void) {
    CHECK(test_direct() == 0);
    CHECK(test_efficientad() == 0);
    CHECK(test_dfkde() == 0);
    CHECK(test_padim() == 0);
    CHECK(test_padim_fitter() == 0);
    CHECK(test_patchcore() == 0);
    CHECK(test_patchcore_fitter() == 0);
    CHECK(test_spade() == 0);
    CHECK(test_yolo() == 0);
    puts("All 9 C ABI object protocol tests passed.");
    return 0;
}
