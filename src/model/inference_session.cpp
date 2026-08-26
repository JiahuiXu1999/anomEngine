#include "model/inference_session.h"

#include "adapters/adapter.h"
#include "backends/backend.h"
#include "backends/backend_factory.h"
#include "processing/postprocessor.h"
#include "processing/preprocessor.h"

#include <algorithm>
#include <chrono>
#include <new>

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

    auto adapter = createAdapter(package.value(), options.pluginDirectory);
    if (!adapter) return adapter.status();

    const auto& manifest = package.value().manifest();
    BackendConfig backendConfig;
    backendConfig.backend = options.backend.value_or(manifest.runtime.backend);
    backendConfig.loadPolicy = options.engineLoadPolicy.value_or(manifest.runtime.loadPolicy);
    backendConfig.fp16 = options.fp16.value_or(manifest.runtime.fp16);
    backendConfig.maxBatchSize = manifest.runtime.maxBatchSize;
    backendConfig.workspaceBytes = manifest.runtime.workspaceBytes;
    backendConfig.inputName = manifest.input.tensorName;
    backendConfig.inputLayout = manifest.input.layout;
    const cv::Size inputSize = manifest.input.centerCrop.value_or(
        cv::Size(manifest.input.resizeWidth, manifest.input.resizeHeight));
    backendConfig.inputHeight = inputSize.height;
    backendConfig.inputWidth = inputSize.width;
    backendConfig.ortProvider = options.ortProvider.value_or(manifest.runtime.ortProvider);
    backendConfig.ortGraphOptimization = manifest.runtime.ortGraphOptimization;
    backendConfig.ortExecutionMode = manifest.runtime.ortExecutionMode;
    backendConfig.ortStrictProvider = manifest.runtime.ortStrictProvider;
    backendConfig.deviceId = manifest.runtime.deviceId;
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

    auto backend = createRuntimeBackend(backendConfig.backend, options.pluginDirectory);
    if (!backend) return backend.status();
    auto loaded = backend.value()->load(backendConfig);
    if (!loaded) return loaded.status();
    auto inputValid = validateInputSignature(manifest, backend.value()->signature());
    if (!inputValid) return inputValid.status();
    auto adapterValid = adapter.value()->validateSignature(backend.value()->signature());
    if (!adapterValid) return adapterValid.status();

    auto session = std::unique_ptr<InferenceSession>(
        new InferenceSession(std::move(package.value()), std::move(backend.value()),
                             std::move(adapter.value())));
    session->modelInfo_.runtimeBackend = backendConfig.backend;
    session->modelInfo_.executionProvider =
        backendConfig.backend == RuntimeBackend::TensorRT
            ? "cuda"
            : toString(backendConfig.ortProvider);
    if (options.warmup) {
        auto warmed = session->warmup();
        if (!warmed) return warmed.status();
    }
    return session;
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
