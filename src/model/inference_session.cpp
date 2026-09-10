#include "model/inference_session.h"

#include "adapters/adapter.h"
#include "backends/backend.h"
#include "backends/backend_factory.h"
#include "processing/postprocessor.h"
#include "processing/preprocessor.h"

#include <algorithm>
#include <chrono>
#include <new>
#include <sstream>
#include <string>
#include <vector>

namespace anom::model {
namespace {

using Clock = std::chrono::steady_clock;

double milliseconds(Clock::time_point begin, Clock::time_point end) {
    return std::chrono::duration<double, std::milli>(end - begin).count();
}

Result<void> validateInputSignature(const ModelManifest& manifest,
                                    const TensorSignature& signature) {
    if (signature.inputs.size() != 1) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Image inference session requires exactly one model input",
                             std::to_string(signature.inputs.size()));
    }
    const auto* input = signature.findInput(manifest.input.tensorName);
    if (!input) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Manifest input tensor is absent from backend",
                             manifest.input.tensorName);
    }
    if (input->dtype != DataType::Float32 || input->shape.dims.size() != 4) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Image input must be a float32 rank-4 tensor",
                             input->name + " " + input->shape.toString());
    }
    if (input->shape.dims[0] > 1) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Only dynamic batch or static batch 1 image inputs are supported",
                             input->shape.toString());
    }
    const cv::Size modelSize = manifest.input.centerCrop.value_or(
        cv::Size(manifest.input.resizeWidth, manifest.input.resizeHeight));
    const int channelAxis = manifest.input.layout == TensorLayout::NCHW ? 1 : 3;
    const int heightAxis = manifest.input.layout == TensorLayout::NCHW ? 2 : 1;
    const int widthAxis = manifest.input.layout == TensorLayout::NCHW ? 3 : 2;
    const auto matches = [](std::int64_t actual, int expected) {
        return actual < 0 || actual == expected;
    };
    if (!matches(input->shape.dims[channelAxis], 3) ||
        !matches(input->shape.dims[heightAxis], modelSize.height) ||
        !matches(input->shape.dims[widthAxis], modelSize.width)) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Manifest preprocessing size/layout does not match backend input",
                             input->shape.toString());
    }
    return {};
}

struct RuntimeCandidate {
    RuntimeBackend backend{RuntimeBackend::OnnxRuntime};
    ExecutionProvider provider{ExecutionProvider::Cpu};
    bool fallback{false};
};

const char* candidateName(const RuntimeCandidate& candidate) noexcept {
    return candidate.backend == RuntimeBackend::TensorRT
        ? "tensorrt/cuda" : "onnxruntime/cpu";
}

bool retryableRuntimeFailure(ErrorCode code) noexcept {
    switch (code) {
        case ErrorCode::ArtifactMissing:
        case ErrorCode::EngineIncompatible:
        case ErrorCode::BackendFailure:
        case ErrorCode::CudaFailure:
        case ErrorCode::PluginNotFound:
        case ErrorCode::PluginAbiMismatch:
        case ErrorCode::DeviceUnavailable:
            return true;
        default:
            return false;
    }
}

std::string joinAttempts(const std::vector<std::string>& attempts) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < attempts.size(); ++i) {
        if (i) stream << " | ";
        stream << attempts[i];
    }
    return stream.str();
}

Result<std::vector<RuntimeCandidate>> runtimeCandidates(
    const BackendConfig& config, const LoadOptions& options) {
    std::vector<RuntimeCandidate> result;
    const auto add = [&](RuntimeBackend backend, ExecutionProvider provider, bool fallback = false) {
        if (options.precision == PrecisionPreference::Float16 &&
            backend != RuntimeBackend::TensorRT) return;
        if (backend == RuntimeBackend::TensorRT &&
            config.enginePath.empty() && config.onnxPath.empty()) return;
        if (backend == RuntimeBackend::OnnxRuntime && config.onnxPath.empty()) return;
        for (const auto& candidate : result) {
            if (candidate.backend == backend && candidate.provider == provider) return;
        }
        result.push_back({backend, provider, fallback});
    };

    const auto constrainedBackend = options.backend;
    switch (options.devicePreference) {
        case DevicePreference::Manifest: {
            const RuntimeBackend backend = constrainedBackend.value_or(config.backend);
            add(backend, backend == RuntimeBackend::TensorRT ? ExecutionProvider::Cuda : ExecutionProvider::Cpu);
            break;
        }
        case DevicePreference::Cpu:
            if (constrainedBackend && *constrainedBackend == RuntimeBackend::TensorRT) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "TensorRT cannot satisfy a CPU execution request");
            }
            add(RuntimeBackend::OnnxRuntime, ExecutionProvider::Cpu);
            break;
        case DevicePreference::Gpu:
            if (!constrainedBackend || *constrainedBackend == RuntimeBackend::TensorRT)
                add(RuntimeBackend::TensorRT, ExecutionProvider::Cuda);
            if (options.fallbackPolicy == FallbackPolicy::LoadOnly &&
                (!constrainedBackend || *constrainedBackend == RuntimeBackend::OnnxRuntime))
                add(RuntimeBackend::OnnxRuntime, ExecutionProvider::Cpu, true);
            break;
        case DevicePreference::Auto:
            if (!constrainedBackend || *constrainedBackend == RuntimeBackend::TensorRT)
                add(RuntimeBackend::TensorRT, ExecutionProvider::Cuda);
            if (!constrainedBackend || *constrainedBackend == RuntimeBackend::OnnxRuntime)
                add(RuntimeBackend::OnnxRuntime, ExecutionProvider::Cpu, true);
            break;
    }

    if (result.empty()) {
        return Status::error(
            ErrorCode::DeviceUnavailable,
            "The model package has no artifact compatible with the requested execution device");
    }
    return result;
}

}  // namespace

InferenceSession::InferenceSession(ModelPackage package,
                                   std::unique_ptr<IRuntimeBackend> backend,
                                   std::unique_ptr<IModelAdapter> adapter)
    : package_(std::move(package)),
      backend_(std::move(backend)),
      adapter_(std::move(adapter)),
      preprocessor_(std::make_unique<ImagePreprocessor>(package_.manifest().input)),
      postprocessor_(std::make_unique<AnomalyPostprocessor>(
          package_.manifest().postprocess,
          package_.manifest().modelId,
          package_.manifest().modelVersion)) {
    modelInfo_.id = package_.manifest().modelId;
    modelInfo_.version = package_.manifest().modelVersion;
    modelInfo_.algorithm = package_.manifest().algorithm;
    modelInfo_.graphContract = package_.manifest().graphContract;
    modelInfo_.signature = backend_->signature();
}

InferenceSession::~InferenceSession() = default;

Result<std::unique_ptr<InferenceSession>> InferenceSession::load(
    const std::filesystem::path& modelPackage, const LoadOptions& options) {
    try {
    auto package = ModelPackage::load(modelPackage);
    if (!package) return package.status();

    const auto& manifest = package.value().manifest();
    BackendConfig backendConfig;
    backendConfig.backend = options.backend.value_or(manifest.runtime.backend);
    backendConfig.loadPolicy = options.engineLoadPolicy.value_or(manifest.runtime.loadPolicy);
    backendConfig.fp16 = options.fp16.value_or(manifest.runtime.fp16);
    if (options.precision == PrecisionPreference::Float32) backendConfig.fp16 = false;
    if (options.precision == PrecisionPreference::Float16) backendConfig.fp16 = true;
    backendConfig.maxBatchSize = manifest.runtime.maxBatchSize;
    backendConfig.workspaceBytes = manifest.runtime.workspaceBytes;
    backendConfig.inputName = manifest.input.tensorName;
    backendConfig.inputLayout = manifest.input.layout;
    const cv::Size inputSize = manifest.input.centerCrop.value_or(
        cv::Size(manifest.input.resizeWidth, manifest.input.resizeHeight));
    backendConfig.inputHeight = inputSize.height;
    backendConfig.inputWidth = inputSize.width;
    backendConfig.ortGraphOptimization = manifest.runtime.ortGraphOptimization;
    backendConfig.ortExecutionMode = manifest.runtime.ortExecutionMode;
    backendConfig.deviceId = options.deviceId.value_or(manifest.runtime.deviceId);
    backendConfig.intraOpThreads = manifest.runtime.intraOpThreads;
    backendConfig.interOpThreads = manifest.runtime.interOpThreads;
    backendConfig.enableMemoryPattern = manifest.runtime.enableMemoryPattern;
    backendConfig.enableCpuMemoryArena = manifest.runtime.enableCpuMemoryArena;
    backendConfig.enableProfiling = manifest.runtime.enableProfiling;
    backendConfig.profileFilePrefix = manifest.runtime.profileFilePrefix;

    if (!manifest.runtime.onnxFile.empty()) {
        auto path = package.value().resolveArtifact(manifest.runtime.onnxFile, true);
        if (!path) return path.status();
        backendConfig.onnxPath = path.value();
    }
    if (!manifest.runtime.engineFile.empty()) {
        const bool mustExist = backendConfig.loadPolicy == EngineLoadPolicy::EngineOnly;
        auto path = package.value().resolveArtifact(manifest.runtime.engineFile, mustExist);
        if (!path) return path.status();
        backendConfig.enginePath = path.value();
    }

    auto candidates = runtimeCandidates(backendConfig, options);
    if (!candidates) return candidates.status();

    std::vector<std::string> attempts;
    Status lastFailure = Status::error(ErrorCode::DeviceUnavailable,
                                       "No runtime candidate was attempted");
    for (const auto& candidate : candidates.value()) {
        BackendConfig selectedConfig = backendConfig;
        selectedConfig.backend = candidate.backend;
        selectedConfig.provider = candidate.provider;
        if (candidate.fallback && attempts.empty()) {
            attempts.push_back(options.devicePreference == DevicePreference::Gpu
                ? "gpu request: no compatible GPU artifact was available"
                : "auto request: no compatible GPU artifact was available");
        }

        auto backend = createRuntimeBackend(candidate.backend, options.pluginDirectory);
        if (!backend) {
            lastFailure = backend.status();
            attempts.push_back(std::string(candidateName(candidate)) + ": " +
                               lastFailure.describe());
            if (!retryableRuntimeFailure(lastFailure.code)) return lastFailure;
            continue;
        }
        auto loaded = backend.value()->load(selectedConfig);
        if (!loaded) {
            lastFailure = loaded.status();
            attempts.push_back(std::string(candidateName(candidate)) + ": " +
                               lastFailure.describe());
            if (!retryableRuntimeFailure(lastFailure.code)) return lastFailure;
            continue;
        }

        auto inputValid = validateInputSignature(manifest, backend.value()->signature());
        if (!inputValid) return inputValid.status();
        auto adapter = createAdapter(package.value(), options.pluginDirectory);
        if (!adapter) return adapter.status();
        auto adapterValid = adapter.value()->validateSignature(backend.value()->signature());
        if (!adapterValid) return adapterValid.status();

        SearchExecutionConfig searchConfig;
        searchConfig.provider = candidate.provider;
        searchConfig.deviceId = selectedConfig.deviceId;
        searchConfig.allowCpuFallback = options.devicePreference == DevicePreference::Auto ||
                                       options.fallbackPolicy == FallbackPolicy::LoadOnly;
        searchConfig.pluginDirectory = options.pluginDirectory;
        auto search = adapter.value()->configureSearch(searchConfig);
        if (!search) {
            lastFailure = search.status();
            attempts.push_back(std::string(candidateName(candidate)) + " Faiss: " + lastFailure.describe());
            if (!retryableSearchFailure(lastFailure.code)) return lastFailure;
            continue;
        }

        auto session = std::unique_ptr<InferenceSession>(
            new InferenceSession(package.value(), std::move(backend.value()),
                                 std::move(adapter.value())));
        session->modelInfo_.runtimeBackend = candidate.backend;
        const bool onGpu = candidate.provider == ExecutionProvider::Cuda;
        session->modelInfo_.executionProvider = onGpu ? "cuda" : "cpu";
        session->executionInfo_.requestedDevice = options.devicePreference;
        session->executionInfo_.runtimeBackend = candidate.backend;
        session->executionInfo_.executionProvider = onGpu ? "cuda" : "cpu";
        session->executionInfo_.deviceId = onGpu ? selectedConfig.deviceId : -1;
        session->executionInfo_.deviceName = onGpu
            ? "CUDA device " + std::to_string(selectedConfig.deviceId) : "CPU";
        session->executionInfo_.precision =
            candidate.backend == RuntimeBackend::TensorRT && selectedConfig.fp16
                ? PrecisionPreference::Float16 : PrecisionPreference::Float32;
        session->executionInfo_.fallbackOccurred = !attempts.empty();
        session->executionInfo_.fallbackReason = joinAttempts(attempts);
        session->executionInfo_.searchProvider = search.value().provider;
        session->executionInfo_.searchDeviceId = search.value().deviceId;
        session->executionInfo_.searchFallbackOccurred = search.value().fallbackOccurred;
        if (search.value().fallbackOccurred) {
            session->executionInfo_.fallbackOccurred = true;
            if (!session->executionInfo_.fallbackReason.empty()) session->executionInfo_.fallbackReason += " | ";
            session->executionInfo_.fallbackReason += "Faiss CUDA -> CPU: " + search.value().fallbackReason;
        }

        if (options.warmup) {
            auto warmed = session->warmup();
            if (!warmed) {
                lastFailure = warmed.status();
                attempts.push_back(std::string(candidateName(candidate)) + " warmup: " +
                                   lastFailure.describe());
                if (!retryableRuntimeFailure(lastFailure.code)) return lastFailure;
                continue;
            }
        }
        return session;
    }

    return Status::error(lastFailure.code,
                         "No compatible runtime candidate could be initialized",
                         joinAttempts(attempts));
    } catch (const std::bad_alloc&) {
        return Status::error(ErrorCode::OutOfMemory, "Out of memory while loading inference session",
                             modelPackage.string());
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::InternalError, "Unexpected exception while loading inference session",
                             error.what());
    }
}

Result<Prediction> InferenceSession::predict(const cv::Mat& image) {
    auto batch = predictBatch({image});
    if (!batch) return batch.status();
    return std::move(batch.value().front());
}

Result<PredictionBatch> InferenceSession::predictBatch(const std::vector<cv::Mat>& images) {
    try {
    if (images.empty()) return Status::error(ErrorCode::InvalidArgument, "Image batch must not be empty");
    PredictionBatch all;
    all.reserve(images.size());
    const std::size_t chunkSize = static_cast<std::size_t>(std::max(1, backend_->maxBatchSize()));
    for (std::size_t begin = 0; begin < images.size(); begin += chunkSize) {
        const std::size_t end = std::min(images.size(), begin + chunkSize);
        std::vector<cv::Mat> chunk(images.begin() + static_cast<std::ptrdiff_t>(begin),
                                   images.begin() + static_cast<std::ptrdiff_t>(end));
        auto predictions = predictChunk(chunk);
        if (!predictions) return predictions.status();
        for (auto& prediction : predictions.value()) all.push_back(std::move(prediction));
    }
    return all;
    } catch (const std::bad_alloc&) {
        return Status::error(ErrorCode::OutOfMemory, "Out of memory during inference");
    } catch (const cv::Exception& error) {
        return Status::error(ErrorCode::InternalError, "OpenCV failed during inference", error.what());
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::InternalError, "Unexpected exception during inference", error.what());
    }
}

Result<PredictionBatch> InferenceSession::predictChunk(const std::vector<cv::Mat>& images) {
    const auto totalBegin = Clock::now();
    auto preprocessed = preprocessor_->process(images);
    if (!preprocessed) return preprocessed.status();
    const auto preprocessEnd = Clock::now();

    TensorMap inputs;
    inputs.emplace(package_.manifest().input.tensorName, std::move(preprocessed.value().tensor));
    const auto backendBegin = Clock::now();
    auto outputs = backend_->infer(inputs);
    if (!outputs) return outputs.status();
    const auto backendEnd = Clock::now();

    const auto adapterBegin = Clock::now();
    auto raw = adapter_->predict(outputs.value(), preprocessor_->modelInputSize());
    if (!raw) return raw.status();
    const auto adapterEnd = Clock::now();

    const auto postprocessBegin = Clock::now();
    auto predictions = postprocessor_->process(raw.value(), preprocessed.value().geometry);
    if (!predictions) return predictions.status();
    const auto postprocessEnd = Clock::now();

    InferenceTiming timing;
    timing.preprocessMs = milliseconds(totalBegin, preprocessEnd);
    timing.backendMs = milliseconds(backendBegin, backendEnd);
    timing.adapterMs = milliseconds(adapterBegin, adapterEnd);
    timing.postprocessMs = milliseconds(postprocessBegin, postprocessEnd);
    timing.totalMs = milliseconds(totalBegin, postprocessEnd);
    for (auto& prediction : predictions.value()) prediction.timing = timing;
    return predictions;
}

Result<void> InferenceSession::warmup() {
    const cv::Size size = preprocessor_->modelInputSize();
    cv::Mat image(size, CV_8UC3, cv::Scalar(0, 0, 0));
    auto result = predict(image);
    if (!result) return result.status();
    return {};
}

}  // namespace anom::model
