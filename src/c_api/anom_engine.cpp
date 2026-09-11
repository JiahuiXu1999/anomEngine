#define ANOM_ENGINE_ENABLE_LEGACY_SESSION_API 1
#include "anomEngine/anomEngine.h"

#include "c_api/algorithm_object_impl.h"
#include "c_api/algorithm_runtime_state.h"

#include "infrastructure/utf8_path.h"

#include "model/inference_session.h"
#include "backends/backend_factory.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <vector>

#ifndef ANOM_ENGINE_VERSION_STRING
#  define ANOM_ENGINE_VERSION_STRING "0.0.0"
#endif

using anom::model::ErrorCode;
using anom::model::AlgorithmType;
using anom::model::DevicePreference;
using anom::model::FallbackPolicy;
using anom::model::InferenceSession;
using anom::model::LoadOptions;
using anom::model::ModelPackage;
using anom::model::pathFromUtf8;
using anom::model::PrecisionPreference;
using anom::model::Prediction;
using anom::model::RuntimeBackend;
using anom::model::Status;

thread_local std::string anomLastError;

using anom::c_api::attachRuntime;
using anom::c_api::runtimeState;

struct anom_session {
    std::unique_ptr<InferenceSession> implementation;
};

namespace {

struct PredictionStorage {
    std::vector<float> rawMap;
    std::vector<float> map;
    std::vector<uint8_t> mask;
    std::vector<anom_region_t> regions;
    std::string modelId;
    std::string modelVersion;
};

anom_status_t publicStatus(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::Ok: return ANOM_STATUS_OK;
        case ErrorCode::InvalidArgument: return ANOM_STATUS_INVALID_ARGUMENT;
        case ErrorCode::InvalidImage: return ANOM_STATUS_INVALID_IMAGE;
        case ErrorCode::IoError:
        case ErrorCode::ArtifactMissing:
        case ErrorCode::ArtifactPathEscape:
        case ErrorCode::ArtifactCorrupt: return ANOM_STATUS_IO_ERROR;
        case ErrorCode::InvalidManifest:
        case ErrorCode::UnsupportedSchema:
        case ErrorCode::TensorSignatureMismatch:
        case ErrorCode::TensorShapeMismatch:
        case ErrorCode::TensorTypeMismatch:
        case ErrorCode::EngineIncompatible: return ANOM_STATUS_INVALID_MODEL_PACKAGE;
        case ErrorCode::UnsupportedAlgorithm: return ANOM_STATUS_PLUGIN_NOT_FOUND;
        case ErrorCode::PluginNotFound: return ANOM_STATUS_PLUGIN_NOT_FOUND;
        case ErrorCode::PluginAbiMismatch: return ANOM_STATUS_PLUGIN_ABI_MISMATCH;
        case ErrorCode::DeviceUnavailable: return ANOM_STATUS_UNSUPPORTED;
        case ErrorCode::OutOfMemory: return ANOM_STATUS_OUT_OF_MEMORY;
        case ErrorCode::BackendFailure:
        case ErrorCode::AdapterFailure:
        case ErrorCode::CudaFailure:
        case ErrorCode::NotInitialized: return ANOM_STATUS_INFERENCE_FAILED;
        case ErrorCode::InternalError: return ANOM_STATUS_INTERNAL_ERROR;
    }
    return ANOM_STATUS_INTERNAL_ERROR;
}

anom_status_t fail(const Status& status) {
    anomLastError = status.describe();
    return publicStatus(status.code);
}

anom_status_t fail(anom_status_t status, std::string message) {
    anomLastError = std::move(message);
    return status;
}

bool validStruct(const void* value, uint32_t size, std::size_t required) {
    return value && size >= required;
}

int channelsFor(anom_pixel_format_t format) {
    switch (format) {
        case ANOM_PIXEL_FORMAT_GRAY8: return 1;
        case ANOM_PIXEL_FORMAT_BGR8:
        case ANOM_PIXEL_FORMAT_RGB8: return 3;
        case ANOM_PIXEL_FORMAT_BGRA8:
        case ANOM_PIXEL_FORMAT_RGBA8: return 4;
        default: return 0;
    }
}

anom_status_t convertImage(const anom_image_t& source, cv::Mat& destination) {
    if (source.struct_size < sizeof(anom_image_t) || !source.data ||
        source.width <= 0 || source.height <= 0) {
        return fail(ANOM_STATUS_INVALID_IMAGE, "Input image descriptor is invalid");
    }
    const int channels = channelsFor(source.pixel_format);
    if (channels == 0) {
        return fail(ANOM_STATUS_INVALID_IMAGE, "Input pixel format is unsupported");
    }
    if (source.width > std::numeric_limits<int32_t>::max() / channels) {
        return fail(ANOM_STATUS_INVALID_IMAGE, "Input image row size overflows");
    }
    const int32_t minimumStride = source.width * channels;
    const int32_t stride = source.stride_bytes == 0 ? minimumStride : source.stride_bytes;
    if (stride < minimumStride) {
        return fail(ANOM_STATUS_INVALID_IMAGE, "Input image stride is too small");
    }
    cv::Mat view(source.height, source.width, CV_MAKETYPE(CV_8U, channels),
                 const_cast<uint8_t*>(source.data), static_cast<std::size_t>(stride));
    if (source.pixel_format == ANOM_PIXEL_FORMAT_RGB8) {
        cv::cvtColor(view, destination, cv::COLOR_RGB2BGR);
    } else if (source.pixel_format == ANOM_PIXEL_FORMAT_RGBA8) {
        cv::cvtColor(view, destination, cv::COLOR_RGBA2BGRA);
    } else {
        destination = view;
    }
    return ANOM_STATUS_OK;
}

template <typename T>
void copyMat(const cv::Mat& source, std::vector<T>& destination) {
    if (source.empty()) return;
    destination.resize(static_cast<std::size_t>(source.rows) * source.cols);
    for (int row = 0; row < source.rows; ++row) {
        std::memcpy(destination.data() + static_cast<std::size_t>(row) * source.cols,
                    source.ptr<T>(row), static_cast<std::size_t>(source.cols) * sizeof(T));
    }
}

anom_status_t exportPrediction(Prediction source, anom_prediction_t* destination) {
    if (!destination || destination->struct_size < sizeof(anom_prediction_t)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "Output prediction struct_size is incompatible");
    }
    if (destination->internal) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "Output prediction must be released before it is reused");
    }
    try {
        auto storage = std::make_unique<PredictionStorage>();
        copyMat<float>(source.rawAnomalyMap, storage->rawMap);
        copyMat<float>(source.anomalyMap, storage->map);
        copyMat<uint8_t>(source.mask, storage->mask);
        storage->regions.reserve(source.analysis.regions.size());
        for (const auto& region : source.analysis.regions) {
            storage->regions.push_back({region.boundingBox.x, region.boundingBox.y,
                                        region.boundingBox.width, region.boundingBox.height,
                                        region.maxLocation.x, region.maxLocation.y, region.area,
                                        region.meanScore, region.maxScore});
        }
        storage->modelId = std::move(source.modelId);
        storage->modelVersion = std::move(source.modelVersion);

        const uint32_t structSize = destination->struct_size;
        std::memset(destination, 0, sizeof(*destination));
        destination->struct_size = structSize;
        destination->raw_score = source.rawScore;
        destination->score = source.score;
        destination->is_anomalous = source.isAnomalous ? 1 : 0;
        destination->raw_anomaly_map = storage->rawMap.empty() ? nullptr : storage->rawMap.data();
        destination->anomaly_map = storage->map.empty() ? nullptr : storage->map.data();
        destination->mask = storage->mask.empty() ? nullptr : storage->mask.data();
        const cv::Mat* spatial = !source.anomalyMap.empty() ? &source.anomalyMap
                              : !source.rawAnomalyMap.empty() ? &source.rawAnomalyMap
                              : !source.mask.empty() ? &source.mask : nullptr;
        if (spatial) {
            destination->map_width = spatial->cols;
            destination->map_height = spatial->rows;
            destination->map_stride_elements = spatial->cols;
            destination->mask_stride_bytes = source.mask.empty() ? 0 : source.mask.cols;
        }
        destination->regions = storage->regions.empty() ? nullptr : storage->regions.data();
        destination->region_count = storage->regions.size();
        destination->anomaly_area_ratio = source.analysis.anomalyAreaRatio;
        destination->mean_score = source.analysis.meanScore;
        destination->max_score = source.analysis.maxScore;
        destination->has_map = source.analysis.hasMap ? 1 : 0;
        destination->model_id_utf8 = storage->modelId.c_str();
        destination->model_version_utf8 = storage->modelVersion.c_str();
        destination->preprocess_ms = source.timing.preprocessMs;
        destination->backend_ms = source.timing.backendMs;
        destination->adapter_ms = source.timing.adapterMs;
        destination->postprocess_ms = source.timing.postprocessMs;
        destination->total_ms = source.timing.totalMs;
        destination->internal = storage.release();
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while exporting prediction");
    }
}

anom_status_t createSession(const char* modelPackage, LoadOptions options,
                            anom_session_t** outSession) {
    if (!outSession) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Session create arguments are null");
    }
    *outSession = nullptr;
    if (!modelPackage) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Session create arguments are null");
    }
    try {
        auto loaded = InferenceSession::load(pathFromUtf8(modelPackage), options);
        if (!loaded) return fail(loaded.status());
        auto session = std::make_unique<anom_session>();
        session->implementation = std::move(loaded.value());
        *outSession = session.release();
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while creating session");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

anom_status_t parseBackend(const char* value, std::optional<RuntimeBackend>& destination) {
    if (!value || !*value) return ANOM_STATUS_OK;
    if (std::strcmp(value, "tensorrt") == 0) {
        destination = RuntimeBackend::TensorRT;
        return ANOM_STATUS_OK;
    }
    if (std::strcmp(value, "onnxruntime") == 0 || std::strcmp(value, "ort") == 0) {
        destination = RuntimeBackend::OnnxRuntime;
        return ANOM_STATUS_OK;
    }
    return fail(ANOM_STATUS_INVALID_ARGUMENT,
                "backend_utf8 must be 'tensorrt' or 'onnxruntime'");
}

anom_status_t setPluginDirectory(const char* value, LoadOptions& destination) {
    if (!value || !*value) return ANOM_STATUS_OK;
    try {
        destination.pluginDirectory = pathFromUtf8(value);
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY,
                    "Out of memory while reading the plugin directory");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, error.what());
    }
}

anom_device_preference_t publicDevice(DevicePreference value) noexcept {
    switch (value) {
        case DevicePreference::Cpu: return ANOM_DEVICE_CPU;
        case DevicePreference::Gpu: return ANOM_DEVICE_GPU;
        case DevicePreference::Manifest:
        case DevicePreference::Auto: return ANOM_DEVICE_AUTO;
    }
    return ANOM_DEVICE_AUTO;
}

anom_precision_t publicPrecision(PrecisionPreference value) noexcept {
    switch (value) {
        case PrecisionPreference::Float32: return ANOM_PRECISION_FP32;
        case PrecisionPreference::Float16: return ANOM_PRECISION_FP16;
        case PrecisionPreference::Manifest:
        case PrecisionPreference::Auto: return ANOM_PRECISION_AUTO;
    }
    return ANOM_PRECISION_AUTO;
}

}  // namespace

extern "C" ANOM_ENGINE_API uint32_t ANOM_CALL anom_get_abi_version(void) {
    return ANOM_ENGINE_ABI_VERSION;
}

extern "C" ANOM_ENGINE_API const char* ANOM_CALL anom_get_version_string(void) {
    return ANOM_ENGINE_VERSION_STRING;
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_create(
    const char* modelPackage, const anom_session_options_t* options,
    anom_session_t** outSession) {
    anomLastError.clear();
    if (outSession) *outSession = nullptr;
    if (options && options->struct_size < sizeof(anom_session_options_t)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "Session options struct_size is incompatible");
    }
    LoadOptions converted;
    if (options) {
        converted.warmup = options->warmup != 0;
        const anom_status_t directoryStatus =
            setPluginDirectory(options->plugin_directory_utf8, converted);
        if (directoryStatus != ANOM_STATUS_OK) return directoryStatus;
    }
    return createSession(modelPackage, std::move(converted), outSession);
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_create_v2(
    const char* modelPackage, const anom_session_options_v2_t* options,
    anom_session_t** outSession) {
    anomLastError.clear();
    if (outSession) *outSession = nullptr;
    constexpr std::size_t minimumSize = offsetof(anom_session_options_v2_t, reserved);
    if (options && options->struct_size < minimumSize) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "Session v2 options struct_size is incompatible");
    }

    LoadOptions converted;
    converted.devicePreference = DevicePreference::Auto;
    converted.deviceId = 0;
    if (options) {
        switch (options->device) {
            case ANOM_DEVICE_AUTO: converted.devicePreference = DevicePreference::Auto; break;
            case ANOM_DEVICE_CPU: converted.devicePreference = DevicePreference::Cpu; break;
            case ANOM_DEVICE_GPU: converted.devicePreference = DevicePreference::Gpu; break;
            default:
                return fail(ANOM_STATUS_INVALID_ARGUMENT,
                            "Session device preference is invalid");
        }
        switch (options->fallback) {
            case ANOM_FALLBACK_NONE: converted.fallbackPolicy = FallbackPolicy::None; break;
            case ANOM_FALLBACK_LOAD_ONLY:
                converted.fallbackPolicy = FallbackPolicy::LoadOnly;
                break;
            default:
                return fail(ANOM_STATUS_INVALID_ARGUMENT,
                            "Session fallback policy is invalid");
        }
        switch (options->precision) {
            case ANOM_PRECISION_AUTO: converted.precision = PrecisionPreference::Auto; break;
            case ANOM_PRECISION_FP32: converted.precision = PrecisionPreference::Float32; break;
            case ANOM_PRECISION_FP16: converted.precision = PrecisionPreference::Float16; break;
            default:
                return fail(ANOM_STATUS_INVALID_ARGUMENT,
                            "Session precision preference is invalid");
        }
        if (options->device_id < -1) {
            return fail(ANOM_STATUS_INVALID_ARGUMENT,
                        "Session device_id must be -1 or non-negative");
        }
        converted.deviceId = options->device_id < 0 ? 0 : options->device_id;
        const anom_status_t backendStatus = parseBackend(options->backend_utf8,
                                                         converted.backend);
        if (backendStatus != ANOM_STATUS_OK) return backendStatus;
        converted.warmup = options->warmup != 0;
        const anom_status_t directoryStatus =
            setPluginDirectory(options->plugin_directory_utf8, converted);
        if (directoryStatus != ANOM_STATUS_OK) return directoryStatus;
    }
    return createSession(modelPackage, std::move(converted), outSession);
}

extern "C" ANOM_ENGINE_API void ANOM_CALL anom_session_destroy(anom_session_t* session) {
    delete session;
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_warmup(anom_session_t* session) {
    anomLastError.clear();
    if (!session) return fail(ANOM_STATUS_INVALID_ARGUMENT, "Session is null");
    try {
        auto warmed = session->implementation->warmup();
        return warmed ? ANOM_STATUS_OK : fail(warmed.status());
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_get_model_info(
    const anom_session_t* session, anom_model_info_t* outInfo) {
    anomLastError.clear();
    if (!session || !outInfo || outInfo->struct_size < sizeof(anom_model_info_t)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Model info arguments are invalid");
    }
    const uint32_t structSize = outInfo->struct_size;
    std::memset(outInfo, 0, sizeof(*outInfo));
    outInfo->struct_size = structSize;
    const auto& info = session->implementation->modelInfo();
    outInfo->model_id_utf8 = info.id.c_str();
    outInfo->model_version_utf8 = info.version.c_str();
    outInfo->algorithm_utf8 = anom::model::toString(info.algorithm);
    outInfo->backend_utf8 = anom::model::toString(info.runtimeBackend);
    outInfo->execution_provider_utf8 = info.executionProvider.c_str();
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_get_execution_info(
    const anom_session_t* session, anom_execution_info_t* outInfo) {
    anomLastError.clear();
    constexpr std::size_t minimumSize = offsetof(anom_execution_info_t, faiss_provider);
    if (!session || !outInfo || outInfo->struct_size < minimumSize) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Execution info arguments are invalid");
    }
    const uint32_t structSize = outInfo->struct_size;
    std::memset(outInfo, 0, std::min<std::size_t>(structSize, sizeof(*outInfo)));
    outInfo->struct_size = structSize;
    const auto& info = session->implementation->executionInfo();
    outInfo->requested_device = publicDevice(info.requestedDevice);
    outInfo->backend_utf8 = anom::model::toString(info.runtimeBackend);
    outInfo->execution_provider_utf8 = info.executionProvider.c_str();
    outInfo->device_id = info.deviceId;
    outInfo->device_name_utf8 = info.deviceName.c_str();
    outInfo->precision = publicPrecision(info.precision);
    outInfo->fallback_occurred = info.fallbackOccurred ? 1 : 0;
    outInfo->fallback_reason_utf8 = info.fallbackReason.c_str();
    if (structSize >= offsetof(anom_execution_info_t, reserved)) {
        outInfo->faiss_provider = static_cast<int32_t>(info.searchProvider);
        outInfo->faiss_device_id = info.searchDeviceId;
        outInfo->faiss_fallback_occurred = info.searchFallbackOccurred ? 1 : 0;
    }
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_predict(
    anom_session_t* session, const anom_image_t* image,
    anom_prediction_t* outPrediction) {
    return anom_session_predict_batch(session, image, 1, outPrediction);
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_session_predict_batch(
    anom_session_t* session, const anom_image_t* images, size_t imageCount,
    anom_prediction_t* outPredictions) {
    anomLastError.clear();
    if (!session || !images || imageCount == 0 || !outPredictions) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Batch prediction arguments are invalid");
    }
    for (std::size_t i = 0; i < imageCount; ++i) {
        if (images[i].struct_size < sizeof(anom_image_t) ||
            outPredictions[i].struct_size < sizeof(anom_prediction_t) ||
            outPredictions[i].internal) {
            return fail(ANOM_STATUS_INVALID_ARGUMENT,
                        "Batch image or prediction descriptor is incompatible");
        }
    }
    try {
        std::vector<cv::Mat> converted;
        converted.reserve(imageCount);
        for (std::size_t i = 0; i < imageCount; ++i) {
            cv::Mat image;
            const anom_status_t status = convertImage(images[i], image);
            if (status != ANOM_STATUS_OK) return status;
            converted.push_back(std::move(image));
        }
        auto predicted = session->implementation->predictBatch(converted);
        if (!predicted) return fail(predicted.status());
        if (predicted.value().size() != imageCount) {
            return fail(ANOM_STATUS_INTERNAL_ERROR,
                        "Inference returned an unexpected prediction count");
        }
        std::size_t exported = 0;
        for (; exported < imageCount; ++exported) {
            const anom_status_t status =
                exportPrediction(std::move(predicted.value()[exported]), &outPredictions[exported]);
            if (status != ANOM_STATUS_OK) {
                for (std::size_t i = 0; i < exported; ++i)
                    anom_prediction_release(&outPredictions[i]);
                return status;
            }
        }
        return ANOM_STATUS_OK;
    } catch (const cv::Exception& error) {
        return fail(ANOM_STATUS_INVALID_IMAGE, error.what());
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory during batch prediction");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

namespace {

template <typename Algorithm>
anom_status_t loadAlgorithm(const char* modelPackage,
                            const anom_algorithm_options_t* options,
                            Algorithm* algorithm,
                            AlgorithmType expected,
                            const char* expectedName) {
    anomLastError.clear();
    if (!algorithm || algorithm->struct_size < sizeof(Algorithm)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    std::string(expectedName) + " object struct_size is incompatible");
    }
    if (algorithm->internal && runtimeState(algorithm)->session) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    std::string(expectedName) + " object is already loaded");
    }
    if (!modelPackage || !*modelPackage) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    std::string(expectedName) + " model package path is empty");
    }
    try {
        auto inspected = ModelPackage::load(pathFromUtf8(modelPackage));
        if (!inspected) return fail(inspected.status());
        if (inspected.value().manifest().algorithm != expected) {
            const std::string actual =
                anom::model::toString(inspected.value().manifest().algorithm);
            return fail(ANOM_STATUS_ALGORITHM_MISMATCH,
                        std::string(expectedName) + " cannot load a " + actual +
                            " model package");
        }
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY,
                    std::string("Out of memory while inspecting ") + expectedName +
                        " model package");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INVALID_MODEL_PACKAGE, error.what());
    }
    anom_session_t* session = nullptr;
    const anom_status_t loaded = anom_session_create_v2(modelPackage, options, &session);
    if (loaded != ANOM_STATUS_OK) return loaded;
    if (!attachRuntime(algorithm->internal, session)) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while creating algorithm state");
    }
    return ANOM_STATUS_OK;
}

template <typename Algorithm>
void releaseAlgorithm(Algorithm* algorithm) {
    if (!algorithm || algorithm->struct_size < sizeof(Algorithm)) return;
    delete runtimeState(algorithm);
    algorithm->internal = nullptr;
    std::memset(algorithm->reserved, 0, sizeof(algorithm->reserved));
}

template <typename Algorithm>
anom_session_t* algorithmSession(Algorithm* algorithm, const char* name) {
    if (!algorithm || algorithm->struct_size < sizeof(Algorithm) ||
        !algorithm->internal || !runtimeState(algorithm)->session) {
        fail(ANOM_STATUS_INVALID_ARGUMENT, std::string(name) + " object is not loaded");
        return nullptr;
    }
    return runtimeState(algorithm)->session;
}

template <typename Algorithm>
const anom_session_t* algorithmSession(const Algorithm* algorithm, const char* name) {
    if (!algorithm || algorithm->struct_size < sizeof(Algorithm) ||
        !algorithm->internal || !runtimeState(algorithm)->session) {
        fail(ANOM_STATUS_INVALID_ARGUMENT, std::string(name) + " object is not loaded");
        return nullptr;
    }
    return runtimeState(algorithm)->session;
}

}  // namespace

namespace anom::c_api {

#define ANOM_DEFINE_ALGORITHM_IMPL(name, type, display_name)                         \
    anom_status_t ANOM_CALL name##Load(                                              \
        anom_##name##_t* algorithm, const char* modelPackage,                       \
        const anom_algorithm_options_t* options) {                                  \
        return loadAlgorithm(modelPackage, options, algorithm, type, display_name); \
    }                                                                                \
    void ANOM_CALL name##Release(anom_##name##_t* algorithm) {                      \
        releaseAlgorithm(algorithm);                                                 \
    }                                                                                \
    anom_status_t ANOM_CALL name##Warmup(anom_##name##_t* algorithm) {              \
        auto* session = algorithmSession(algorithm, display_name);                   \
        return session ? anom_session_warmup(session)                               \
                       : ANOM_STATUS_INVALID_ARGUMENT;                               \
    }                                                                                \
    anom_status_t ANOM_CALL name##GetModelInfo(                                     \
        const anom_##name##_t* algorithm, anom_model_info_t* outInfo) {             \
        const auto* session = algorithmSession(algorithm, display_name);             \
        return session ? anom_session_get_model_info(session, outInfo)              \
                       : ANOM_STATUS_INVALID_ARGUMENT;                               \
    }                                                                                \
    anom_status_t ANOM_CALL name##GetExecutionInfo(                                 \
        const anom_##name##_t* algorithm, anom_execution_info_t* outInfo) {         \
        const auto* session = algorithmSession(algorithm, display_name);             \
        return session ? anom_session_get_execution_info(session, outInfo)          \
                       : ANOM_STATUS_INVALID_ARGUMENT;                               \
    }                                                                                \
    anom_status_t ANOM_CALL name##Predict(                                          \
        anom_##name##_t* algorithm, const anom_image_t* image,                      \
        anom_prediction_t* outPrediction) {                                         \
        auto* session = algorithmSession(algorithm, display_name);                   \
        return session ? anom_session_predict(session, image, outPrediction)        \
                       : ANOM_STATUS_INVALID_ARGUMENT;                               \
    }                                                                                \
    anom_status_t ANOM_CALL name##PredictBatch(                                     \
        anom_##name##_t* algorithm, const anom_image_t* images,                     \
        size_t imageCount, anom_prediction_t* outPredictions) {                     \
        auto* session = algorithmSession(algorithm, display_name);                   \
        return session ? anom_session_predict_batch(                                \
                             session, images, imageCount, outPredictions)            \
                       : ANOM_STATUS_INVALID_ARGUMENT;                               \
    }

ANOM_DEFINE_ALGORITHM_IMPL(direct, AlgorithmType::Direct, "Direct")
ANOM_DEFINE_ALGORITHM_IMPL(efficientad, AlgorithmType::EfficientAD, "EfficientAD")
ANOM_DEFINE_ALGORITHM_IMPL(dfkde, AlgorithmType::DFKDE, "DFKDE")
ANOM_DEFINE_ALGORITHM_IMPL(padim, AlgorithmType::Padim, "PaDiM")
ANOM_DEFINE_ALGORITHM_IMPL(patchcore, AlgorithmType::PatchCore, "PatchCore")
ANOM_DEFINE_ALGORITHM_IMPL(spade, AlgorithmType::SPADE, "SPADE")
ANOM_DEFINE_ALGORITHM_IMPL(yolo, AlgorithmType::Yolo, "YOLO")

#undef ANOM_DEFINE_ALGORITHM_IMPL

}  // namespace anom::c_api

extern "C" ANOM_ENGINE_API void ANOM_CALL anom_prediction_release(anom_prediction_t* prediction) {
    if (!prediction) return;
    delete static_cast<PredictionStorage*>(prediction->internal);
    const uint32_t structSize = prediction->struct_size;
    std::memset(prediction, 0, sizeof(*prediction));
    prediction->struct_size = structSize;
}

extern "C" ANOM_ENGINE_API size_t ANOM_CALL anom_get_last_error(char* buffer, size_t bufferSize) {
    const std::size_t required = anomLastError.size() + 1;
    if (buffer && bufferSize != 0) {
        const std::size_t count = std::min(anomLastError.size(), bufferSize - 1);
        std::memcpy(buffer, anomLastError.data(), count);
        buffer[count] = '\0';
    }
    return required;
}

extern "C" ANOM_ENGINE_API anom_status_t ANOM_CALL anom_runtime_probe(
    const char* backendName, const char* providerName, int32_t deviceId,
    const char* pluginDirectory) {
    anomLastError.clear();
    try {
        if (!backendName || !*backendName || !providerName || deviceId < -1)
            return fail(ANOM_STATUS_INVALID_ARGUMENT, "Runtime probe arguments are invalid");
        std::optional<RuntimeBackend> backend;
        const auto parsed = parseBackend(backendName, backend);
        if (parsed != ANOM_STATUS_OK) return parsed;
        anom::model::ExecutionProvider provider;
        if (std::strcmp(providerName, "cpu") == 0) provider = anom::model::ExecutionProvider::Cpu;
        else if (std::strcmp(providerName, "cuda") == 0) provider = anom::model::ExecutionProvider::Cuda;
        else return fail(ANOM_STATUS_INVALID_ARGUMENT, "Provider must be cpu or cuda");
        if (*backend == RuntimeBackend::TensorRT && provider == anom::model::ExecutionProvider::Cpu)
            return fail(ANOM_STATUS_INVALID_ARGUMENT, "TensorRT requires the CUDA provider");
        if (*backend == RuntimeBackend::OnnxRuntime && provider == anom::model::ExecutionProvider::Cuda)
            return fail(ANOM_STATUS_INVALID_ARGUMENT, "ONNX Runtime requires the CPU provider");
        LoadOptions options;
        const auto directory = setPluginDirectory(pluginDirectory, options);
        if (directory != ANOM_STATUS_OK) return directory;
        auto runtime = anom::model::createRuntimeBackend(*backend, options.pluginDirectory);
        if (!runtime) return fail(runtime.status());
        auto available = runtime.value()->probe(provider, deviceId < 0 ? 0 : deviceId);
        return available ? ANOM_STATUS_OK : fail(available.status());
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while probing runtime");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}
