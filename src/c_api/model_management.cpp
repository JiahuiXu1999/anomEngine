#define ANOM_ENGINE_ENABLE_LEGACY_SESSION_API 1
#include "anomEngine/anomEngine.h"

#ifndef ANOM_ENGINE_HAS_PATCHCORE
#  define ANOM_ENGINE_HAS_PATCHCORE 0
#endif
#ifndef ANOM_ENGINE_HAS_PADIM
#  define ANOM_ENGINE_HAS_PADIM 0
#endif
#ifndef ANOM_ENGINE_HAS_DIRECT
#  define ANOM_ENGINE_HAS_DIRECT 0
#endif
#ifndef ANOM_ENGINE_HAS_EFFICIENTAD
#  define ANOM_ENGINE_HAS_EFFICIENTAD 0
#endif
#ifndef ANOM_ENGINE_HAS_DFKDE
#  define ANOM_ENGINE_HAS_DFKDE 0
#endif
#ifndef ANOM_ENGINE_HAS_SPADE
#  define ANOM_ENGINE_HAS_SPADE 0
#endif
#ifndef ANOM_ENGINE_HAS_YOLO
#  define ANOM_ENGINE_HAS_YOLO 0
#endif

#include "backends/backend.h"
#include "backends/backend_factory.h"
#include "infrastructure/json.h"
#include "infrastructure/sha256.h"
#include "infrastructure/utf8_path.h"
#include "model/inference_session.h"
#include "model/manifest.h"
#if ANOM_ENGINE_HAS_PADIM
#  include "pipeline/padim.h"
#endif
#if ANOM_ENGINE_HAS_PATCHCORE
#  include "pipeline/patchcore.h"
#endif
#include "processing/preprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

using anom::model::AlgorithmType;
using anom::model::BackendConfig;
using anom::model::EngineLoadPolicy;
using anom::model::ErrorCode;
using anom::model::ImagePreprocessor;
using anom::model::InferenceSession;
using anom::model::IRuntimeBackend;
using anom::model::ModelPackage;
using anom::model::PadimConfig;
using anom::model::PatchCoreConfig;
#if ANOM_ENGINE_HAS_PADIM
using anom::model::PadimPipeline;
using anom::model::PadimPipelineConfig;
#endif
#if ANOM_ENGINE_HAS_PATCHCORE
using anom::model::PatchCorePipeline;
using anom::model::PatchCorePipelineConfig;
#endif
using anom::model::Result;
using anom::model::RuntimeBackend;
using anom::model::Status;
using anom::model::TensorMap;
using anom::model::pathFromUtf8;
using anom::model::pathToUtf8;

extern thread_local std::string anomLastError;

struct anom_model {
    explicit anom_model(ModelPackage value) : package(std::move(value)) {
        const auto& manifest = package.manifest();
        modelId = manifest.modelId;
        modelVersion = manifest.modelVersion;
        algorithm = anom::model::toString(manifest.algorithm);
        backend = anom::model::toString(manifest.runtime.backend);
        executionProvider = manifest.runtime.backend == RuntimeBackend::TensorRT
                                ? "cuda"
                                : "cpu";
    }

    ModelPackage package;
    std::string modelId;
    std::string modelVersion;
    std::string algorithm;
    std::string backend;
    std::string executionProvider;
};

struct ArtifactOverlay {
    std::filesystem::path source;
    std::filesystem::path relative;
};

struct CalibrationData {
    float imageMin{0.0F};
    float imageMax{1.0F};
    float imageThreshold{0.5F};
    float pixelMin{0.0F};
    float pixelMax{1.0F};
    float pixelThreshold{0.5F};
    bool hasPixelStatistics{false};
};

struct anom_package_builder {
    std::filesystem::path templateRoot;
    std::vector<ArtifactOverlay> artifacts;
    std::optional<CalibrationData> calibration;
    bool committed{false};
};

struct anom_fitter {
    enum class Kind { PatchCore, Padim };

    ModelPackage package;
    std::filesystem::path pluginDirectory;
    std::unique_ptr<IRuntimeBackend> backend;
    std::unique_ptr<ImagePreprocessor> preprocessor;
#if ANOM_ENGINE_HAS_PATCHCORE
    std::unique_ptr<PatchCorePipeline> patchcore;
#endif
#if ANOM_ENGINE_HAS_PADIM
    std::unique_ptr<PadimPipeline> padim;
#endif
    Kind kind{Kind::PatchCore};
    std::atomic<anom_fit_stage_t> stage{ANOM_FIT_STAGE_READY};
    std::atomic<std::uint64_t> processedSamples{0};
    std::atomic<std::uint64_t> collectedItems{0};
    std::atomic<bool> cancellationRequested{false};

    anom_fitter(ModelPackage value, std::filesystem::path plugin)
        : package(std::move(value)), pluginDirectory(std::move(plugin)) {}
};

struct anom_calibrator {
    std::unique_ptr<InferenceSession> session;
    float targetFalsePositiveRate{0.01F};
    std::size_t maxPixelSamples{1'000'000};
    std::vector<float> imageScores;
    std::vector<float> pixelScores;
    std::uint64_t seenPixels{0};
    std::mt19937_64 random{42};
};

namespace {

struct CapabilityEntry {
    const char* name;
    anom_algorithm_capabilities_t capabilities;
    bool available;
};

constexpr anom_algorithm_capabilities_t kImported =
    ANOM_ALGORITHM_CAP_PACKAGE_IMPORT | ANOM_ALGORITHM_CAP_CALIBRATION;
constexpr anom_algorithm_capabilities_t kFitted =
    ANOM_ALGORITHM_CAP_PACKAGE_IMPORT |
    ANOM_ALGORITHM_CAP_ARTIFACT_FIT |
    ANOM_ALGORITHM_CAP_INCREMENTAL_FIT |
    ANOM_ALGORITHM_CAP_CALIBRATION;

constexpr CapabilityEntry kCapabilities[] = {
    {"patchcore", kFitted, ANOM_ENGINE_HAS_PATCHCORE != 0},
    {"padim", kFitted, ANOM_ENGINE_HAS_PADIM != 0},
    {"direct", kImported, ANOM_ENGINE_HAS_DIRECT != 0},
    {"efficientad", kImported, ANOM_ENGINE_HAS_EFFICIENTAD != 0},
    {"dfkde", kImported, ANOM_ENGINE_HAS_DFKDE != 0},
    {"spade", kImported, ANOM_ENGINE_HAS_SPADE != 0},
    {"yolo", kImported, ANOM_ENGINE_HAS_YOLO != 0},
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
        case ErrorCode::UnsupportedAlgorithm: return ANOM_STATUS_UNSUPPORTED;
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

bool safeRelativePath(const std::filesystem::path& path) {
    if (path.empty() || path.is_absolute()) return false;
    const auto normalized = path.lexically_normal();
    if (normalized.empty() || normalized == ".") return false;
    for (const auto& component : normalized) {
        if (component == "..") return false;
    }
    return true;
}

std::string genericUtf8(const std::filesystem::path& path) {
    const std::u8string encoded = path.generic_u8string();
    std::string result(encoded.size(), '\0');
    if (!encoded.empty()) std::memcpy(result.data(), encoded.data(), encoded.size());
    return result;
}

bool pathIsInside(const std::filesystem::path& child,
                  const std::filesystem::path& parent) {
    const auto relative = child.lexically_relative(parent);
    if (relative.empty()) return child == parent;
    for (const auto& component : relative) {
        if (component == "..") return false;
        break;
    }
    return !relative.is_absolute();
}

std::filesystem::path uniqueSibling(const std::filesystem::path& output,
                                    const char* tag) {
    static std::atomic<std::uint64_t> counter{0};
    const auto ticks = static_cast<std::uint64_t>(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto sequence = counter.fetch_add(1, std::memory_order_relaxed);
    return output.parent_path() /
           ("." + output.filename().string() + "." + tag + "-" +
            std::to_string(ticks) + "-" + std::to_string(sequence));
}

struct DirectoryCleanup {
    std::filesystem::path path;
    bool active{true};
    ~DirectoryCleanup() {
        if (!active || path.empty()) return;
        std::error_code ignored;
        std::filesystem::remove_all(path, ignored);
    }
};

Result<void> copyTemplateTree(const std::filesystem::path& source,
                              const std::filesystem::path& destination) {
    try {
        std::filesystem::create_directories(destination);
        for (const auto& entry : std::filesystem::recursive_directory_iterator(source)) {
            if (entry.is_symlink()) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "Package templates must not contain symbolic links",
                                     entry.path().string());
            }
            const auto relative = std::filesystem::relative(entry.path(), source);
            const auto target = destination / relative;
            if (entry.is_directory()) {
                std::filesystem::create_directories(target);
            } else if (entry.is_regular_file()) {
                std::filesystem::create_directories(target.parent_path());
                std::filesystem::copy_file(entry.path(), target,
                                           std::filesystem::copy_options::overwrite_existing);
            }
        }
        return {};
    } catch (const std::filesystem::filesystem_error& error) {
        return Status::error(ErrorCode::IoError,
                             "Unable to copy package template", error.what());
    }
}

Result<void> writeChecksummedManifest(const std::filesystem::path& root,
                                      const CalibrationData* calibration) {
    const auto manifestPath = root / "manifest.json";
    std::ifstream stream(manifestPath, std::ios::binary);
    if (!stream) {
        return Status::error(ErrorCode::ArtifactMissing,
                             "Package template manifest is missing", manifestPath.string());
    }
    const std::string text((std::istreambuf_iterator<char>(stream)),
                           std::istreambuf_iterator<char>());
    auto document = anom::model::json::parse(text);
    if (!document) return document.status();
    if (!document.value().isObject()) {
        return Status::error(ErrorCode::InvalidManifest,
                             "Model manifest root must be an object");
    }

    if (calibration) {
        auto& rootObject = document.value().asObject();
        auto found = rootObject.find("postprocess");
        if (found == rootObject.end() || !found->second.isObject()) {
            found = rootObject.emplace(
                "postprocess", anom::model::json::Value(
                    anom::model::json::Value::Object{})).first;
        }
        auto& postprocess = found->second.asObject();
        postprocess["normalize"] = anom::model::json::Value(true);
        postprocess["threshold"] = anom::model::json::Value(true);
        postprocess["image_min"] =
            anom::model::json::Value(static_cast<double>(calibration->imageMin));
        postprocess["image_max"] =
            anom::model::json::Value(static_cast<double>(calibration->imageMax));
        postprocess["image_threshold"] =
            anom::model::json::Value(static_cast<double>(calibration->imageThreshold));
        if (calibration->hasPixelStatistics) {
            postprocess["pixel_min"] =
                anom::model::json::Value(static_cast<double>(calibration->pixelMin));
            postprocess["pixel_max"] =
                anom::model::json::Value(static_cast<double>(calibration->pixelMax));
            postprocess["pixel_threshold"] =
                anom::model::json::Value(static_cast<double>(calibration->pixelThreshold));
        }
    }

    anom::model::json::Value::Object checksums;
    try {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root)) {
            if (entry.is_symlink()) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "Model packages must not contain symbolic links",
                                     entry.path().string());
            }
            if (!entry.is_regular_file() || entry.path() == manifestPath) continue;
            const auto relative = std::filesystem::relative(entry.path(), root);
            auto digest = anom::model::sha256File(entry.path());
            if (!digest) return digest.status();
            checksums.emplace(genericUtf8(relative),
                              anom::model::json::Value(std::move(digest.value())));
        }
    } catch (const std::filesystem::filesystem_error& error) {
        return Status::error(ErrorCode::IoError,
                             "Unable to enumerate package artifacts", error.what());
    }
    document.value().asObject()["checksums"] =
        anom::model::json::Value(std::move(checksums));

    std::ofstream output(manifestPath, std::ios::binary | std::ios::trunc);
    if (!output) {
        return Status::error(ErrorCode::IoError,
                             "Unable to rewrite package manifest", manifestPath.string());
    }
    output << anom::model::json::serialize(document.value(), true);
    if (!output) {
        return Status::error(ErrorCode::IoError,
                             "Unable to write package manifest", manifestPath.string());
    }
    return {};
}

Result<void> validateReferencedArtifacts(const ModelPackage& package) {
    const auto& manifest = package.manifest();
    if (!manifest.runtime.onnxFile.empty()) {
        auto artifact = package.resolveArtifact(manifest.runtime.onnxFile, true);
        if (!artifact) return artifact.status();
    }
    if (manifest.runtime.loadPolicy == EngineLoadPolicy::EngineOnly) {
        auto artifact = package.resolveArtifact(manifest.runtime.engineFile, true);
        if (!artifact) return artifact.status();
    }
    if (manifest.algorithm == AlgorithmType::PatchCore) {
        const auto& config = std::get<PatchCoreConfig>(manifest.algorithmConfig);
        auto artifact = package.resolveArtifact(config.indexFile, true);
        if (!artifact) return artifact.status();
    } else if (manifest.algorithm == AlgorithmType::Padim) {
        const auto& config = std::get<PadimConfig>(manifest.algorithmConfig);
        auto statistics = package.resolveArtifact(config.statisticsFile, true);
        if (!statistics) return statistics.status();
        auto channels = package.resolveArtifact(config.channelIndicesFile, true);
        if (!channels) return channels.status();
    } else if (manifest.algorithm == AlgorithmType::DFKDE) {
        const auto& config = std::get<anom::model::DFKDEConfig>(manifest.algorithmConfig);
        auto artifact = package.resolveArtifact(config.statisticsFile, true);
        if (!artifact) return artifact.status();
    } else if (manifest.algorithm == AlgorithmType::SPADE) {
        const auto& config = std::get<anom::model::SPADEConfig>(manifest.algorithmConfig);
        for (const auto& index : config.indexFiles) {
            auto artifact = package.resolveArtifact(index, true);
            if (!artifact) return artifact.status();
        }
    }
    return {};
}

Result<void> commitPackage(const std::filesystem::path& templateRoot,
                           const std::vector<ArtifactOverlay>& artifacts,
                           const std::filesystem::path& requestedOutput,
                           const CalibrationData* calibration = nullptr) {
    try {
        if (requestedOutput.empty() || !requestedOutput.has_filename()) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "Output package path is invalid", requestedOutput.string());
        }
        const auto output = std::filesystem::absolute(requestedOutput).lexically_normal();
        const auto source = std::filesystem::weakly_canonical(templateRoot);
        const auto outputParent = output.parent_path();
        std::filesystem::create_directories(outputParent);
        const auto canonicalParent = std::filesystem::weakly_canonical(outputParent);
        const auto prospectiveOutput = canonicalParent / output.filename();
        if (pathIsInside(prospectiveOutput, source)) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "Output package must not be inside its template package",
                                 prospectiveOutput.string());
        }
        if (std::filesystem::exists(prospectiveOutput)) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "Output package already exists", prospectiveOutput.string());
        }

        const auto staging = uniqueSibling(prospectiveOutput, "anomtmp");
        DirectoryCleanup cleanup{staging};
        auto copied = copyTemplateTree(source, staging);
        if (!copied) return copied.status();
        for (const auto& artifact : artifacts) {
            if (!safeRelativePath(artifact.relative) ||
                artifact.relative.lexically_normal() == "manifest.json") {
                return Status::error(ErrorCode::InvalidArgument,
                                     "Artifact destination is not a safe package-relative path",
                                     artifact.relative.string());
            }
            if (!std::filesystem::is_regular_file(artifact.source)) {
                return Status::error(ErrorCode::ArtifactMissing,
                                     "Package artifact source does not exist",
                                     artifact.source.string());
            }
            const auto destination = staging / artifact.relative.lexically_normal();
            std::filesystem::create_directories(destination.parent_path());
            std::filesystem::copy_file(artifact.source, destination,
                                       std::filesystem::copy_options::overwrite_existing);
        }

        auto checksummed = writeChecksummedManifest(staging, calibration);
        if (!checksummed) return checksummed.status();
        auto package = ModelPackage::load(staging);
        if (!package) return package.status();
        auto artifactsValid = validateReferencedArtifacts(package.value());
        if (!artifactsValid) return artifactsValid.status();

        std::filesystem::rename(staging, prospectiveOutput);
        cleanup.active = false;
        return {};
    } catch (const std::filesystem::filesystem_error& error) {
        return Status::error(ErrorCode::IoError,
                             "Unable to commit model package", error.what());
    }
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
    if (channels == 0 || source.width > std::numeric_limits<std::int32_t>::max() / channels) {
        return fail(ANOM_STATUS_INVALID_IMAGE, "Input pixel format or dimensions are invalid");
    }
    const std::int32_t minimumStride = source.width * channels;
    const std::int32_t stride = source.stride_bytes == 0 ? minimumStride : source.stride_bytes;
    if (stride < minimumStride) {
        return fail(ANOM_STATUS_INVALID_IMAGE, "Input image stride is too small");
    }
    cv::Mat view(source.height, source.width, CV_MAKETYPE(CV_8U, channels),
                 const_cast<std::uint8_t*>(source.data), static_cast<std::size_t>(stride));
    if (source.pixel_format == ANOM_PIXEL_FORMAT_RGB8) {
        cv::cvtColor(view, destination, cv::COLOR_RGB2BGR);
    } else if (source.pixel_format == ANOM_PIXEL_FORMAT_RGBA8) {
        cv::cvtColor(view, destination, cv::COLOR_RGBA2BGRA);
    } else {
        destination = view;
    }
    return ANOM_STATUS_OK;
}

Result<std::unique_ptr<IRuntimeBackend>> createFeatureBackend(
    const ModelPackage& package, const std::filesystem::path& pluginDirectory) {
    const auto& manifest = package.manifest();
    BackendConfig config;
    config.backend = manifest.runtime.backend;
    config.loadPolicy = manifest.runtime.loadPolicy;
    config.fp16 = manifest.runtime.fp16;
    config.maxBatchSize = manifest.runtime.maxBatchSize;
    config.workspaceBytes = manifest.runtime.workspaceBytes;
    config.inputName = manifest.input.tensorName;
    config.inputLayout = manifest.input.layout;
    const cv::Size inputSize = manifest.input.centerCrop.value_or(
        cv::Size(manifest.input.resizeWidth, manifest.input.resizeHeight));
    config.inputHeight = inputSize.height;
    config.inputWidth = inputSize.width;
    config.ortGraphOptimization = manifest.runtime.ortGraphOptimization;
    config.ortExecutionMode = manifest.runtime.ortExecutionMode;
    config.deviceId = manifest.runtime.deviceId;
    config.intraOpThreads = manifest.runtime.intraOpThreads;
    config.interOpThreads = manifest.runtime.interOpThreads;
    config.enableMemoryPattern = manifest.runtime.enableMemoryPattern;
    config.enableCpuMemoryArena = manifest.runtime.enableCpuMemoryArena;
    config.enableProfiling = manifest.runtime.enableProfiling;
    config.profileFilePrefix = manifest.runtime.profileFilePrefix;
    if (!manifest.runtime.onnxFile.empty()) {
        auto path = package.resolveArtifact(manifest.runtime.onnxFile, true);
        if (!path) return path.status();
        config.onnxPath = path.value();
    }
    if (!manifest.runtime.engineFile.empty()) {
        auto path = package.resolveArtifact(
            manifest.runtime.engineFile,
            manifest.runtime.loadPolicy == EngineLoadPolicy::EngineOnly);
        if (!path) return path.status();
        config.enginePath = path.value();
    }
    auto backend = anom::model::createRuntimeBackend(config.backend, pluginDirectory);
    if (!backend) return backend.status();
    auto loaded = backend.value()->load(config);
    if (!loaded) return loaded.status();
    return std::move(backend.value());
}

void fillModelInfo(const anom_model& model, anom_model_info_t* output) {
    const std::uint32_t structSize = output->struct_size;
    std::memset(output, 0, sizeof(*output));
    output->struct_size = structSize;
    output->model_id_utf8 = model.modelId.c_str();
    output->model_version_utf8 = model.modelVersion.c_str();
    output->algorithm_utf8 = model.algorithm.c_str();
    output->backend_utf8 = model.backend.c_str();
    output->execution_provider_utf8 = model.executionProvider.c_str();
}

}  // namespace

extern "C" ANOM_ENGINE_API size_t anom_algorithm_get_count(void) {
    return sizeof(kCapabilities) / sizeof(kCapabilities[0]);
}

extern "C" ANOM_ENGINE_API anom_status_t anom_algorithm_get_info(
    size_t index, anom_algorithm_info_t* outInfo) {
    anomLastError.clear();
    if (!outInfo || outInfo->struct_size < sizeof(anom_algorithm_info_t) ||
        index >= anom_algorithm_get_count()) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "Algorithm capability query arguments are invalid");
    }
    const auto& entry = kCapabilities[index];
    const std::uint32_t structSize = outInfo->struct_size;
    std::memset(outInfo, 0, sizeof(*outInfo));
    outInfo->struct_size = structSize;
    outInfo->algorithm_utf8 = entry.name;
    outInfo->capabilities = entry.capabilities;
    outInfo->available = entry.available ? 1 : 0;
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_model_open(
    const char* modelPackage, anom_model_t** outModel) {
    anomLastError.clear();
    if (!outModel) return fail(ANOM_STATUS_INVALID_ARGUMENT, "Model output pointer is null");
    *outModel = nullptr;
    if (!modelPackage || !*modelPackage) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Model package path is empty");
    }
    try {
        auto package = ModelPackage::load(pathFromUtf8(modelPackage));
        if (!package) return fail(package.status());
        auto model = std::make_unique<anom_model>(std::move(package.value()));
        *outModel = model.release();
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while opening model package");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_model_get_info(
    const anom_model_t* model, anom_model_info_t* outInfo) {
    anomLastError.clear();
    if (!model || !outInfo || outInfo->struct_size < sizeof(anom_model_info_t)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Model info arguments are invalid");
    }
    fillModelInfo(*model, outInfo);
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_model_validate(
    const char* modelPackage, const anom_model_validate_options_t* options,
    anom_model_validation_report_t* outReport) {
    anomLastError.clear();
    if (!modelPackage || !*modelPackage || !outReport ||
        outReport->struct_size < sizeof(anom_model_validation_report_t) ||
        (options && options->struct_size < sizeof(anom_model_validate_options_t))) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Model validation arguments are invalid");
    }
    const std::uint32_t structSize = outReport->struct_size;
    std::memset(outReport, 0, sizeof(*outReport));
    outReport->struct_size = structSize;
    try {
        auto package = ModelPackage::load(pathFromUtf8(modelPackage));
        if (!package) return fail(package.status());
        auto referenced = validateReferencedArtifacts(package.value());
        if (!referenced) return fail(referenced.status());
        outReport->package_valid = 1;
        if (options && options->initialize_runtime) {
            anom::model::LoadOptions loadOptions;
            if (options->plugin_directory_utf8 && *options->plugin_directory_utf8) {
                loadOptions.pluginDirectory = pathFromUtf8(options->plugin_directory_utf8);
            }
            auto session = InferenceSession::load(pathFromUtf8(modelPackage), loadOptions);
            if (!session) return fail(session.status());
            outReport->runtime_valid = 1;
        }
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while validating model package");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API void anom_model_destroy(anom_model_t* model) {
    delete model;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_package_builder_create(
    const anom_package_builder_options_t* options,
    anom_package_builder_t** outBuilder) {
    anomLastError.clear();
    if (!outBuilder) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Package builder output pointer is null");
    }
    *outBuilder = nullptr;
    if (!options || options->struct_size < sizeof(anom_package_builder_options_t) ||
        !options->template_package_utf8 || !*options->template_package_utf8) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Package builder options are invalid");
    }
    try {
        auto package = ModelPackage::load(pathFromUtf8(options->template_package_utf8));
        if (!package) return fail(package.status());
        auto builder = std::make_unique<anom_package_builder>();
        builder->templateRoot = package.value().root();
        *outBuilder = builder.release();
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while creating package builder");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_package_builder_add_artifact(
    anom_package_builder_t* builder, const char* sourcePath,
    const char* packageRelativePath) {
    anomLastError.clear();
    if (!builder || builder->committed || !sourcePath || !*sourcePath ||
        !packageRelativePath || !*packageRelativePath) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Package artifact arguments are invalid");
    }
    try {
        const auto source = pathFromUtf8(sourcePath);
        const auto relative = pathFromUtf8(packageRelativePath).lexically_normal();
        if (!std::filesystem::is_regular_file(source)) {
            return fail(ANOM_STATUS_IO_ERROR, "Package artifact source does not exist");
        }
        if (!safeRelativePath(relative) || relative == "manifest.json") {
            return fail(ANOM_STATUS_INVALID_ARGUMENT,
                        "Artifact destination must be a safe package-relative path");
        }
        const auto duplicate = std::find_if(
            builder->artifacts.begin(), builder->artifacts.end(),
            [&](const ArtifactOverlay& value) { return value.relative == relative; });
        if (duplicate != builder->artifacts.end()) {
            duplicate->source = source;
        } else {
            builder->artifacts.push_back({source, relative});
        }
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while adding package artifact");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_package_builder_commit(
    anom_package_builder_t* builder, const char* outputPackage) {
    anomLastError.clear();
    if (!builder || builder->committed || !outputPackage || !*outputPackage) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Package commit arguments are invalid");
    }
    auto committed = commitPackage(builder->templateRoot, builder->artifacts,
                                   pathFromUtf8(outputPackage),
                                   builder->calibration ? &*builder->calibration : nullptr);
    if (!committed) return fail(committed.status());
    builder->committed = true;
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_package_builder_apply_calibration(
    anom_package_builder_t* builder,
    const anom_calibration_result_t* calibration) {
    anomLastError.clear();
    if (!builder || builder->committed || !calibration ||
        calibration->struct_size < sizeof(anom_calibration_result_t) ||
        calibration->sample_count == 0 ||
        !std::isfinite(calibration->image_min) ||
        !std::isfinite(calibration->image_max) ||
        !std::isfinite(calibration->image_threshold) ||
        calibration->image_max <= calibration->image_min ||
        (calibration->has_pixel_statistics &&
         (!std::isfinite(calibration->pixel_min) ||
          !std::isfinite(calibration->pixel_max) ||
          !std::isfinite(calibration->pixel_threshold) ||
          calibration->pixel_max <= calibration->pixel_min))) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Calibration result is invalid");
    }
    builder->calibration = CalibrationData{
        calibration->image_min,
        calibration->image_max,
        calibration->image_threshold,
        calibration->pixel_min,
        calibration->pixel_max,
        calibration->pixel_threshold,
        calibration->has_pixel_statistics != 0};
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API void anom_package_builder_destroy(
    anom_package_builder_t* builder) {
    delete builder;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_create(
    const anom_fitter_options_t* options, anom_fitter_t** outFitter) {
    anomLastError.clear();
    if (!outFitter) return fail(ANOM_STATUS_INVALID_ARGUMENT, "Fitter output pointer is null");
    *outFitter = nullptr;
    if (!options || options->struct_size < sizeof(anom_fitter_options_t) ||
        !options->template_package_utf8 || !*options->template_package_utf8 ||
        (!options->padim_channel_indices && options->padim_channel_index_count != 0)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Fitter options are invalid");
    }
    try {
        auto loaded = ModelPackage::load(pathFromUtf8(options->template_package_utf8));
        if (!loaded) return fail(loaded.status());
        const auto algorithm = loaded.value().manifest().algorithm;
        if (algorithm != AlgorithmType::PatchCore && algorithm != AlgorithmType::Padim) {
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "This algorithm imports a trained graph and does not use artifact fitting");
        }
        if ((algorithm == AlgorithmType::PatchCore && !ANOM_ENGINE_HAS_PATCHCORE) ||
            (algorithm == AlgorithmType::Padim && !ANOM_ENGINE_HAS_PADIM)) {
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "Artifact fitting for this algorithm is not available in this build");
        }

        std::filesystem::path pluginDirectory;
        if (options->plugin_directory_utf8 && *options->plugin_directory_utf8) {
            pluginDirectory = pathFromUtf8(options->plugin_directory_utf8);
        }
        auto fitter = std::make_unique<anom_fitter>(std::move(loaded.value()),
                                                    std::move(pluginDirectory));
        auto backend = createFeatureBackend(fitter->package, fitter->pluginDirectory);
        if (!backend) return fail(backend.status());
        fitter->backend = std::move(backend.value());
        fitter->preprocessor =
            std::make_unique<ImagePreprocessor>(fitter->package.manifest().input);

        if (algorithm == AlgorithmType::PatchCore) {
#if ANOM_ENGINE_HAS_PATCHCORE
            const auto& config =
                std::get<PatchCoreConfig>(fitter->package.manifest().algorithmConfig);
            PatchCorePipelineConfig fitConfig;
            fitConfig.embeddingDimension = config.embeddingDimension;
            fitConfig.coresetSamplingRatio =
                options->patchcore_coreset_sampling_ratio > 0.0F
                    ? options->patchcore_coreset_sampling_ratio : 0.1F;
            fitConfig.projectionDimension =
                options->patchcore_projection_dimension != 0
                    ? options->patchcore_projection_dimension : 128;
            if (options->patchcore_max_exact_distance_evaluations >
                std::numeric_limits<std::size_t>::max()) {
                return fail(ANOM_STATUS_INVALID_ARGUMENT,
                            "PatchCore distance evaluation limit exceeds size_t");
            }
            fitConfig.maxExactDistanceEvaluations =
                options->patchcore_max_exact_distance_evaluations != 0
                    ? static_cast<std::size_t>(
                          options->patchcore_max_exact_distance_evaluations)
                    : 50'000'000;
            fitConfig.randomSeed = options->random_seed != 0 ? options->random_seed : 42;
            fitter->patchcore = std::make_unique<PatchCorePipeline>(fitConfig);
            fitter->kind = anom_fitter::Kind::PatchCore;
            auto existing = fitter->package.resolveArtifact(config.indexFile, false);
            if (!existing) return fail(existing.status());
            if (std::filesystem::is_regular_file(existing.value())) {
                auto imported = fitter->patchcore->loadMemoryBank(existing.value());
                if (!imported) return fail(imported.status());
                fitter->collectedItems.store(imported.value(), std::memory_order_relaxed);
            }
#else
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PatchCore fitting is not available in this build");
#endif
        } else {
#if ANOM_ENGINE_HAS_PADIM
            const auto& config =
                std::get<PadimConfig>(fitter->package.manifest().algorithmConfig);
            PadimPipelineConfig fitConfig;
            fitConfig.embeddingDimension = config.embeddingDimension;
            fitConfig.featureHeight = config.featureHeight;
            fitConfig.featureWidth = config.featureWidth;
            fitConfig.covarianceRegularization =
                options->padim_covariance_regularization > 0.0
                    ? options->padim_covariance_regularization : 0.01;
            if (options->padim_channel_index_count != 0) {
                if (options->padim_channel_index_count !=
                    static_cast<std::size_t>(config.embeddingDimension)) {
                    return fail(ANOM_STATUS_INVALID_ARGUMENT,
                                "PaDiM channel index count must equal embedding dimension");
                }
                fitConfig.channelIndices.assign(
                    options->padim_channel_indices,
                    options->padim_channel_indices + options->padim_channel_index_count);
            } else {
                fitConfig.channelIndices.resize(
                    static_cast<std::size_t>(config.embeddingDimension));
                std::iota(fitConfig.channelIndices.begin(), fitConfig.channelIndices.end(), 0);
            }
            fitter->padim = std::make_unique<PadimPipeline>(std::move(fitConfig));
            fitter->kind = anom_fitter::Kind::Padim;
#else
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PaDiM fitting is not available in this build");
#endif
        }
        *outFitter = fitter.release();
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while creating fitter");
    } catch (const cv::Exception& error) {
        return fail(ANOM_STATUS_INVALID_MODEL_PACKAGE, error.what());
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_add_batch(
    anom_fitter_t* fitter, const anom_image_t* images, size_t imageCount) {
    anomLastError.clear();
    if (!fitter || !images || imageCount == 0 ||
        fitter->stage.load(std::memory_order_acquire) == ANOM_FIT_STAGE_COMPLETE) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Fitter batch arguments are invalid");
    }
    if (fitter->cancellationRequested.load(std::memory_order_acquire)) {
        fitter->stage.store(ANOM_FIT_STAGE_CANCELLED, std::memory_order_release);
        return fail(ANOM_STATUS_CANCELLED, "Fitter cancellation was requested");
    }
    try {
        std::vector<cv::Mat> converted;
        converted.reserve(imageCount);
        for (std::size_t index = 0; index < imageCount; ++index) {
            cv::Mat image;
            const auto status = convertImage(images[index], image);
            if (status != ANOM_STATUS_OK) return status;
            converted.push_back(std::move(image));
        }

        fitter->stage.store(ANOM_FIT_STAGE_EXTRACTING, std::memory_order_release);
        const std::size_t chunkSize = static_cast<std::size_t>(
            std::max(1, fitter->backend->maxBatchSize()));
        for (std::size_t begin = 0; begin < converted.size(); begin += chunkSize) {
            if (fitter->cancellationRequested.load(std::memory_order_acquire)) {
                fitter->stage.store(ANOM_FIT_STAGE_CANCELLED, std::memory_order_release);
                return fail(ANOM_STATUS_CANCELLED, "Fitter cancellation was requested");
            }
            const std::size_t end = std::min(converted.size(), begin + chunkSize);
            std::vector<cv::Mat> chunk(converted.begin() + static_cast<std::ptrdiff_t>(begin),
                                       converted.begin() + static_cast<std::ptrdiff_t>(end));
            auto preprocessed = fitter->preprocessor->process(chunk);
            if (!preprocessed) {
                fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
                return fail(preprocessed.status());
            }
            TensorMap inputs;
            inputs.emplace(fitter->package.manifest().input.tensorName,
                           std::move(preprocessed.value().tensor));
            auto outputs = fitter->backend->infer(inputs);
            if (!outputs) {
                fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
                return fail(outputs.status());
            }
            Result<void> added;
            if (fitter->kind == anom_fitter::Kind::PatchCore) {
#if ANOM_ENGINE_HAS_PATCHCORE
                const auto& config = std::get<PatchCoreConfig>(
                    fitter->package.manifest().algorithmConfig);
                added = fitter->patchcore->addBackendOutputs(
                    outputs.value(), fitter->package.manifest().outputs, config);
#else
                return fail(ANOM_STATUS_UNSUPPORTED,
                            "PatchCore fitting is not available in this build");
#endif
            } else {
#if ANOM_ENGINE_HAS_PADIM
                const auto& config = std::get<PadimConfig>(
                    fitter->package.manifest().algorithmConfig);
                added = fitter->padim->addBackendOutputs(
                    outputs.value(), fitter->package.manifest().outputs, config);
#else
                return fail(ANOM_STATUS_UNSUPPORTED,
                            "PaDiM fitting is not available in this build");
#endif
            }
            if (!added) {
                fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
                return fail(added.status());
            }
            fitter->processedSamples.fetch_add(end - begin, std::memory_order_relaxed);
            if (fitter->kind == anom_fitter::Kind::PatchCore) {
#if ANOM_ENGINE_HAS_PATCHCORE
                fitter->collectedItems.store(fitter->patchcore->featureCount(),
                                             std::memory_order_relaxed);
#endif
            } else {
#if ANOM_ENGINE_HAS_PADIM
                fitter->collectedItems.store(fitter->padim->sampleCount(),
                                             std::memory_order_relaxed);
#endif
            }
        }
        fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
        return ANOM_STATUS_OK;
    } catch (const cv::Exception& error) {
        fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
        return fail(ANOM_STATUS_INVALID_IMAGE, error.what());
    } catch (const std::bad_alloc&) {
        fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while fitting image batch");
    } catch (const std::exception& error) {
        fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_get_progress(
    const anom_fitter_t* fitter, anom_fit_progress_t* outProgress) {
    anomLastError.clear();
    if (!fitter || !outProgress || outProgress->struct_size < sizeof(anom_fit_progress_t)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Fitter progress arguments are invalid");
    }
    const auto structSize = outProgress->struct_size;
    std::memset(outProgress, 0, sizeof(*outProgress));
    outProgress->struct_size = structSize;
    outProgress->stage = fitter->stage.load(std::memory_order_acquire);
    outProgress->processed_samples =
        fitter->processedSamples.load(std::memory_order_relaxed);
    outProgress->collected_items =
        fitter->collectedItems.load(std::memory_order_relaxed);
    outProgress->cancellation_requested =
        fitter->cancellationRequested.load(std::memory_order_acquire) ? 1 : 0;
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_save_checkpoint(
    const anom_fitter_t* fitter, const char* checkpointPath) {
    anomLastError.clear();
    if (!fitter || !checkpointPath || !*checkpointPath) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Checkpoint save arguments are invalid");
    }
    try {
        const auto path = pathFromUtf8(checkpointPath);
        if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
        Result<void> saved;
        if (fitter->kind == anom_fitter::Kind::PatchCore) {
#if ANOM_ENGINE_HAS_PATCHCORE
            saved = fitter->patchcore->saveCheckpoint(path);
#else
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PatchCore fitting is not available in this build");
#endif
        } else {
#if ANOM_ENGINE_HAS_PADIM
            saved = fitter->padim->saveCheckpoint(path);
#else
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PaDiM fitting is not available in this build");
#endif
        }
        return saved ? ANOM_STATUS_OK : fail(saved.status());
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_load_checkpoint(
    anom_fitter_t* fitter, const char* checkpointPath) {
    anomLastError.clear();
    if (!fitter || !checkpointPath || !*checkpointPath ||
        fitter->stage.load(std::memory_order_acquire) != ANOM_FIT_STAGE_READY) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Checkpoint load arguments are invalid");
    }
    try {
        const auto path = pathFromUtf8(checkpointPath);
        Result<void> loaded;
        if (fitter->kind == anom_fitter::Kind::PatchCore) {
#if ANOM_ENGINE_HAS_PATCHCORE
            loaded = fitter->patchcore->loadCheckpoint(path);
#else
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PatchCore fitting is not available in this build");
#endif
        } else {
#if ANOM_ENGINE_HAS_PADIM
            loaded = fitter->padim->loadCheckpoint(path);
#else
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PaDiM fitting is not available in this build");
#endif
        }
        if (!loaded) return fail(loaded.status());
        if (fitter->kind == anom_fitter::Kind::PatchCore) {
            /* The PatchCore checkpoint stores patch vectors, not source-image
             * provenance. A non-zero marker keeps the resumed fitter
             * finalizable while collected_items remains exact. */
            fitter->processedSamples.store(1, std::memory_order_relaxed);
#if ANOM_ENGINE_HAS_PATCHCORE
            fitter->collectedItems.store(fitter->patchcore->featureCount(),
                                         std::memory_order_relaxed);
#endif
        } else {
#if ANOM_ENGINE_HAS_PADIM
            fitter->processedSamples.store(fitter->padim->sampleCount(),
                                           std::memory_order_relaxed);
            fitter->collectedItems.store(fitter->padim->sampleCount(),
                                         std::memory_order_relaxed);
#endif
        }
        return ANOM_STATUS_OK;
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_cancel(anom_fitter_t* fitter) {
    anomLastError.clear();
    if (!fitter) return fail(ANOM_STATUS_INVALID_ARGUMENT, "Fitter is null");
    fitter->cancellationRequested.store(true, std::memory_order_release);
    fitter->stage.store(ANOM_FIT_STAGE_CANCELLED, std::memory_order_release);
    return ANOM_STATUS_OK;
}

extern "C" ANOM_ENGINE_API anom_status_t anom_fitter_finalize(
    anom_fitter_t* fitter, const char* outputPackage) {
    anomLastError.clear();
    if (!fitter || !outputPackage || !*outputPackage ||
        fitter->processedSamples.load(std::memory_order_relaxed) == 0 ||
        fitter->stage.load(std::memory_order_acquire) != ANOM_FIT_STAGE_READY) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Fitter finalize arguments or state are invalid");
    }
    if (fitter->cancellationRequested.load(std::memory_order_acquire)) {
        return fail(ANOM_STATUS_CANCELLED, "Fitter cancellation was requested");
    }
    fitter->stage.store(ANOM_FIT_STAGE_FINALIZING, std::memory_order_release);
    try {
        const auto output = std::filesystem::absolute(pathFromUtf8(outputPackage));
        std::filesystem::create_directories(output.parent_path());
        const auto artifactRoot = uniqueSibling(output, "fit");
        DirectoryCleanup cleanup{artifactRoot};
        std::filesystem::create_directories(artifactRoot);
        std::vector<ArtifactOverlay> artifacts;
        if (fitter->kind == anom_fitter::Kind::PatchCore) {
#if ANOM_ENGINE_HAS_PATCHCORE
            const auto& config = std::get<PatchCoreConfig>(
                fitter->package.manifest().algorithmConfig);
            const auto artifact = artifactRoot / config.indexFile;
            std::filesystem::create_directories(artifact.parent_path());
            auto saved = fitter->patchcore->saveMemoryBank(artifact);
            if (!saved) {
                fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
                return fail(saved.status());
            }
            fitter->collectedItems.store(saved.value(), std::memory_order_relaxed);
            artifacts.push_back({artifact, config.indexFile});
#else
            fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PatchCore fitting is not available in this build");
#endif
        } else {
#if ANOM_ENGINE_HAS_PADIM
            const auto& config = std::get<PadimConfig>(
                fitter->package.manifest().algorithmConfig);
            const auto statistics = artifactRoot / config.statisticsFile;
            const auto channels = artifactRoot / config.channelIndicesFile;
            std::filesystem::create_directories(statistics.parent_path());
            std::filesystem::create_directories(channels.parent_path());
            auto saved = fitter->padim->saveArtifacts(statistics, channels);
            if (!saved) {
                fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
                return fail(saved.status());
            }
            artifacts.push_back({statistics, config.statisticsFile});
            artifacts.push_back({channels, config.channelIndicesFile});
#else
            fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
            return fail(ANOM_STATUS_UNSUPPORTED,
                        "PaDiM fitting is not available in this build");
#endif
        }
        auto committed = commitPackage(fitter->package.root(), artifacts, output);
        if (!committed) {
            fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
            return fail(committed.status());
        }
        fitter->stage.store(ANOM_FIT_STAGE_COMPLETE, std::memory_order_release);
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while finalizing fitter");
    } catch (const std::exception& error) {
        fitter->stage.store(ANOM_FIT_STAGE_READY, std::memory_order_release);
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API void anom_fitter_destroy(anom_fitter_t* fitter) {
    delete fitter;
}

namespace {

template <typename Fitter>
anom_fitter_t* typedFitter(Fitter* fitter, const char* name) {
    if (!fitter || fitter->struct_size < sizeof(Fitter) || !fitter->internal) {
        fail(ANOM_STATUS_INVALID_ARGUMENT, std::string(name) + " fitter is not initialized");
        return nullptr;
    }
    return static_cast<anom_fitter_t*>(fitter->internal);
}

template <typename Fitter>
const anom_fitter_t* typedFitter(const Fitter* fitter, const char* name) {
    if (!fitter || fitter->struct_size < sizeof(Fitter) || !fitter->internal) {
        fail(ANOM_STATUS_INVALID_ARGUMENT, std::string(name) + " fitter is not initialized");
        return nullptr;
    }
    return static_cast<const anom_fitter_t*>(fitter->internal);
}

template <typename Fitter>
anom_status_t attachTypedFitter(Fitter* destination,
                                anom_fitter_t* implementation,
                                anom_fitter::Kind expected,
                                const char* name) {
    if (implementation->kind != expected) {
        anom_fitter_destroy(implementation);
        return fail(ANOM_STATUS_INVALID_MODEL_PACKAGE,
                    std::string(name) + " fitter requires a matching model template");
    }
    destination->internal = implementation;
    return ANOM_STATUS_OK;
}

anom_status_t validateFitterTemplate(const char* templatePackage,
                                     AlgorithmType expected,
                                     const char* name) {
    if (!templatePackage || !*templatePackage) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    std::string(name) + " fitter template path is empty");
    }
    try {
        auto inspected = ModelPackage::load(pathFromUtf8(templatePackage));
        if (!inspected) return fail(inspected.status());
        if (inspected.value().manifest().algorithm != expected) {
            return fail(ANOM_STATUS_ALGORITHM_MISMATCH,
                        std::string(name) + " fitter cannot use a " +
                            toString(inspected.value().manifest().algorithm) +
                            " model template");
        }
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY,
                    std::string("Out of memory while inspecting ") + name +
                        " fitter template");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INVALID_MODEL_PACKAGE, error.what());
    }
}

}  // namespace

extern "C" ANOM_ENGINE_API anom_status_t anom_patchcore_fitter_create(
    const anom_patchcore_fitter_options_t* options,
    anom_patchcore_fitter_t* fitter) {
    anomLastError.clear();
    if (!fitter || fitter->struct_size < sizeof(*fitter) || fitter->internal ||
        !options || options->struct_size < sizeof(*options)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "PatchCore fitter object or options are invalid");
    }
    const auto validated = validateFitterTemplate(
        options->template_package_utf8, AlgorithmType::PatchCore, "PatchCore");
    if (validated != ANOM_STATUS_OK) return validated;
    anom_fitter_options_t converted{};
    converted.struct_size = sizeof(converted);
    converted.template_package_utf8 = options->template_package_utf8;
    converted.plugin_directory_utf8 = options->plugin_directory_utf8;
    converted.patchcore_coreset_sampling_ratio = options->coreset_sampling_ratio;
    converted.patchcore_projection_dimension = options->projection_dimension;
    converted.patchcore_max_exact_distance_evaluations =
        options->max_exact_distance_evaluations;
    converted.random_seed = options->random_seed;
    anom_fitter_t* implementation = nullptr;
    const auto created = anom_fitter_create(&converted, &implementation);
    if (created != ANOM_STATUS_OK) return created;
    return attachTypedFitter(fitter, implementation, anom_fitter::Kind::PatchCore,
                             "PatchCore");
}

extern "C" ANOM_ENGINE_API anom_status_t anom_padim_fitter_create(
    const anom_padim_fitter_options_t* options,
    anom_padim_fitter_t* fitter) {
    anomLastError.clear();
    if (!fitter || fitter->struct_size < sizeof(*fitter) || fitter->internal ||
        !options || options->struct_size < sizeof(*options)) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT,
                    "PaDiM fitter object or options are invalid");
    }
    const auto validated = validateFitterTemplate(
        options->template_package_utf8, AlgorithmType::Padim, "PaDiM");
    if (validated != ANOM_STATUS_OK) return validated;
    anom_fitter_options_t converted{};
    converted.struct_size = sizeof(converted);
    converted.template_package_utf8 = options->template_package_utf8;
    converted.plugin_directory_utf8 = options->plugin_directory_utf8;
    converted.padim_covariance_regularization = options->covariance_regularization;
    converted.padim_channel_indices = options->channel_indices;
    converted.padim_channel_index_count = options->channel_index_count;
    anom_fitter_t* implementation = nullptr;
    const auto created = anom_fitter_create(&converted, &implementation);
    if (created != ANOM_STATUS_OK) return created;
    return attachTypedFitter(fitter, implementation, anom_fitter::Kind::Padim,
                             "PaDiM");
}

#define ANOM_DEFINE_FITTER_API(name, display_name)                                      \
    extern "C" ANOM_ENGINE_API anom_status_t anom_##name##_fitter_add_batch(          \
        anom_##name##_fitter_t* fitter, const anom_image_t* images, size_t imageCount) {\
        auto* implementation = typedFitter(fitter, display_name);                       \
        return implementation ? anom_fitter_add_batch(implementation, images, imageCount)\
                              : ANOM_STATUS_INVALID_ARGUMENT;                           \
    }                                                                                   \
    extern "C" ANOM_ENGINE_API anom_status_t anom_##name##_fitter_get_progress(       \
        const anom_##name##_fitter_t* fitter, anom_fit_progress_t* outProgress) {       \
        const auto* implementation = typedFitter(fitter, display_name);                 \
        return implementation ? anom_fitter_get_progress(implementation, outProgress)  \
                              : ANOM_STATUS_INVALID_ARGUMENT;                           \
    }                                                                                   \
    extern "C" ANOM_ENGINE_API anom_status_t anom_##name##_fitter_save_checkpoint(    \
        const anom_##name##_fitter_t* fitter, const char* path) {                       \
        const auto* implementation = typedFitter(fitter, display_name);                 \
        return implementation ? anom_fitter_save_checkpoint(implementation, path)      \
                              : ANOM_STATUS_INVALID_ARGUMENT;                           \
    }                                                                                   \
    extern "C" ANOM_ENGINE_API anom_status_t anom_##name##_fitter_load_checkpoint(    \
        anom_##name##_fitter_t* fitter, const char* path) {                             \
        auto* implementation = typedFitter(fitter, display_name);                       \
        return implementation ? anom_fitter_load_checkpoint(implementation, path)      \
                              : ANOM_STATUS_INVALID_ARGUMENT;                           \
    }                                                                                   \
    extern "C" ANOM_ENGINE_API anom_status_t anom_##name##_fitter_cancel(             \
        anom_##name##_fitter_t* fitter) {                                               \
        auto* implementation = typedFitter(fitter, display_name);                       \
        return implementation ? anom_fitter_cancel(implementation)                     \
                              : ANOM_STATUS_INVALID_ARGUMENT;                           \
    }                                                                                   \
    extern "C" ANOM_ENGINE_API anom_status_t anom_##name##_fitter_finalize(           \
        anom_##name##_fitter_t* fitter, const char* outputPackage) {                    \
        auto* implementation = typedFitter(fitter, display_name);                       \
        return implementation ? anom_fitter_finalize(implementation, outputPackage)    \
                              : ANOM_STATUS_INVALID_ARGUMENT;                           \
    }                                                                                   \
    extern "C" ANOM_ENGINE_API void anom_##name##_fitter_release(                     \
        anom_##name##_fitter_t* fitter) {                                               \
        if (!fitter) return;                                                            \
        anom_fitter_destroy(static_cast<anom_fitter_t*>(fitter->internal));             \
        fitter->internal = nullptr;                                                     \
        std::memset(fitter->reserved, 0, sizeof(fitter->reserved));                     \
    }

ANOM_DEFINE_FITTER_API(patchcore, "PatchCore")
ANOM_DEFINE_FITTER_API(padim, "PaDiM")

#undef ANOM_DEFINE_FITTER_API

extern "C" ANOM_ENGINE_API anom_status_t anom_calibrator_create(
    const anom_model_t* model, const anom_calibrator_options_t* options,
    anom_calibrator_t** outCalibrator) {
    anomLastError.clear();
    if (!outCalibrator) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Calibrator output pointer is null");
    }
    *outCalibrator = nullptr;
    if (!model || (options && options->struct_size < sizeof(anom_calibrator_options_t))) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Calibrator create arguments are invalid");
    }
    try {
        anom::model::LoadOptions loadOptions;
        float targetFalsePositiveRate = 0.01F;
        std::uint64_t maxPixelSamples = 1'000'000;
        std::uint32_t randomSeed = 42;
        if (options) {
            if (options->plugin_directory_utf8 && *options->plugin_directory_utf8) {
                loadOptions.pluginDirectory = pathFromUtf8(options->plugin_directory_utf8);
            }
            if (options->target_false_positive_rate != 0.0F) {
                targetFalsePositiveRate = options->target_false_positive_rate;
            }
            if (options->max_pixel_samples != 0) {
                maxPixelSamples = options->max_pixel_samples;
            }
            if (options->random_seed != 0) randomSeed = options->random_seed;
        }
        if (!std::isfinite(targetFalsePositiveRate) ||
            targetFalsePositiveRate <= 0.0F || targetFalsePositiveRate >= 1.0F ||
            maxPixelSamples > std::numeric_limits<std::size_t>::max()) {
            return fail(ANOM_STATUS_INVALID_ARGUMENT, "Calibrator options are invalid");
        }
        auto session = InferenceSession::load(model->package.root(), loadOptions);
        if (!session) return fail(session.status());
        auto calibrator = std::make_unique<anom_calibrator>();
        calibrator->session = std::move(session.value());
        calibrator->targetFalsePositiveRate = targetFalsePositiveRate;
        calibrator->maxPixelSamples = static_cast<std::size_t>(maxPixelSamples);
        calibrator->random.seed(randomSeed);
        calibrator->pixelScores.reserve(
            std::min<std::size_t>(calibrator->maxPixelSamples, 65'536));
        *outCalibrator = calibrator.release();
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while creating calibrator");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_calibrator_add_batch(
    anom_calibrator_t* calibrator, const anom_image_t* images, size_t imageCount) {
    anomLastError.clear();
    if (!calibrator || !images || imageCount == 0) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Calibrator batch arguments are invalid");
    }
    try {
        std::vector<cv::Mat> converted;
        converted.reserve(imageCount);
        for (std::size_t index = 0; index < imageCount; ++index) {
            cv::Mat image;
            const auto status = convertImage(images[index], image);
            if (status != ANOM_STATUS_OK) return status;
            converted.push_back(std::move(image));
        }
        auto predictions = calibrator->session->predictBatch(converted);
        if (!predictions) return fail(predictions.status());
        for (const auto& prediction : predictions.value()) {
            if (!std::isfinite(prediction.rawScore)) {
                return fail(ANOM_STATUS_INFERENCE_FAILED,
                            "Calibration prediction contains a non-finite image score");
            }
            calibrator->imageScores.push_back(prediction.rawScore);
            if (prediction.rawAnomalyMap.empty()) continue;
            for (int row = 0; row < prediction.rawAnomalyMap.rows; ++row) {
                const float* values = prediction.rawAnomalyMap.ptr<float>(row);
                for (int column = 0; column < prediction.rawAnomalyMap.cols; ++column) {
                    const float value = values[column];
                    if (!std::isfinite(value)) {
                        return fail(ANOM_STATUS_INFERENCE_FAILED,
                                    "Calibration prediction contains a non-finite pixel score");
                    }
                    ++calibrator->seenPixels;
                    if (calibrator->pixelScores.size() < calibrator->maxPixelSamples) {
                        calibrator->pixelScores.push_back(value);
                    } else {
                        std::uniform_int_distribution<std::uint64_t> select(
                            0, calibrator->seenPixels - 1);
                        const std::uint64_t selected = select(calibrator->random);
                        if (selected < calibrator->maxPixelSamples) {
                            calibrator->pixelScores[static_cast<std::size_t>(selected)] = value;
                        }
                    }
                }
            }
        }
        return ANOM_STATUS_OK;
    } catch (const cv::Exception& error) {
        return fail(ANOM_STATUS_INVALID_IMAGE, error.what());
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while calibrating image batch");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API anom_status_t anom_calibrator_compute(
    const anom_calibrator_t* calibrator, anom_calibration_result_t* outResult) {
    anomLastError.clear();
    if (!calibrator || !outResult ||
        outResult->struct_size < sizeof(anom_calibration_result_t) ||
        calibrator->imageScores.empty()) {
        return fail(ANOM_STATUS_INVALID_ARGUMENT, "Calibrator compute arguments or state are invalid");
    }
    try {
        const auto quantile = [](std::vector<float> values, float targetFalsePositiveRate) {
            std::sort(values.begin(), values.end());
            const double q = 1.0 - static_cast<double>(targetFalsePositiveRate);
            const std::size_t index = static_cast<std::size_t>(
                std::floor(q * static_cast<double>(values.size() - 1)));
            return values[index];
        };
        const auto range = [](const std::vector<float>& values) {
            const auto bounds = std::minmax_element(values.begin(), values.end());
            float minimum = *bounds.first;
            float maximum = *bounds.second;
            if (maximum <= minimum) {
                const float delta = std::max(1.0e-6F, std::abs(minimum) * 1.0e-6F);
                minimum -= delta;
                maximum += delta;
            }
            return std::pair<float, float>{minimum, maximum};
        };

        const auto imageRange = range(calibrator->imageScores);
        const auto structSize = outResult->struct_size;
        std::memset(outResult, 0, sizeof(*outResult));
        outResult->struct_size = structSize;
        outResult->sample_count = calibrator->imageScores.size();
        outResult->image_min = imageRange.first;
        outResult->image_max = imageRange.second;
        outResult->image_threshold =
            quantile(calibrator->imageScores, calibrator->targetFalsePositiveRate);
        outResult->pixel_min = 0.0F;
        outResult->pixel_max = 1.0F;
        outResult->pixel_threshold = 0.5F;
        if (!calibrator->pixelScores.empty()) {
            const auto pixelRange = range(calibrator->pixelScores);
            outResult->pixel_min = pixelRange.first;
            outResult->pixel_max = pixelRange.second;
            outResult->pixel_threshold =
                quantile(calibrator->pixelScores, calibrator->targetFalsePositiveRate);
            outResult->has_pixel_statistics = 1;
        }
        return ANOM_STATUS_OK;
    } catch (const std::bad_alloc&) {
        return fail(ANOM_STATUS_OUT_OF_MEMORY, "Out of memory while computing calibration");
    } catch (const std::exception& error) {
        return fail(ANOM_STATUS_INTERNAL_ERROR, error.what());
    }
}

extern "C" ANOM_ENGINE_API void anom_calibrator_destroy(
    anom_calibrator_t* calibrator) {
    delete calibrator;
}
