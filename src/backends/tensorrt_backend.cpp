#include "backends/tensorrt_backend.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <unordered_map>
#include <utility>
#include <vector>

#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#endif

namespace anom::model {
namespace {

class TensorRTLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT][" << severityName(severity) << "] " << message << '\n';
        }
    }

private:
    static const char* severityName(Severity severity) noexcept {
        switch (severity) {
            case Severity::kINTERNAL_ERROR: return "FATAL";
            case Severity::kERROR: return "ERROR";
            case Severity::kWARNING: return "WARN";
            case Severity::kINFO: return "INFO";
            case Severity::kVERBOSE: return "VERBOSE";
        }
        return "UNKNOWN";
    }
};

DataType convertType(nvinfer1::DataType type) {
    switch (type) {
        case nvinfer1::DataType::kFLOAT: return DataType::Float32;
        case nvinfer1::DataType::kHALF: return DataType::Float16;
        case nvinfer1::DataType::kINT32: return DataType::Int32;
        case nvinfer1::DataType::kINT64: return DataType::Int64;
        case nvinfer1::DataType::kUINT8: return DataType::UInt8;
        case nvinfer1::DataType::kBOOL: return DataType::Bool;
        default: throw std::runtime_error("Unsupported TensorRT tensor data type");
    }
}

TensorShape convertShape(const nvinfer1::Dims& dims) {
    TensorShape shape;
    shape.dims.reserve(static_cast<std::size_t>(dims.nbDims));
    for (int i = 0; i < dims.nbDims; ++i) shape.dims.push_back(dims.d[i]);
    return shape;
}

Result<nvinfer1::Dims> convertShape(const TensorShape& shape) {
    if (shape.dims.size() > static_cast<std::size_t>(nvinfer1::Dims::MAX_DIMS)) {
        return Status::error(ErrorCode::TensorShapeMismatch, "Tensor rank exceeds TensorRT limit",
                             shape.toString());
    }
    nvinfer1::Dims dims{};
    dims.nbDims = static_cast<int>(shape.dims.size());
    for (int i = 0; i < dims.nbDims; ++i) {
        const auto value = shape.dims[static_cast<std::size_t>(i)];
        if (value <= 0 || value > std::numeric_limits<std::int32_t>::max()) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Runtime tensor dimensions must be positive int32 values", shape.toString());
        }
        dims.d[i] = static_cast<std::int32_t>(value);
    }
    return dims;
}

Status cudaStatus(cudaError_t error, const std::string& operation) {
    return Status::error(error == cudaErrorMemoryAllocation ? ErrorCode::OutOfMemory : ErrorCode::CudaFailure,
                         operation + " failed", cudaGetErrorString(error));
}

std::vector<char> readBinary(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("Unable to open " + path.string());
    const auto end = stream.tellg();
    if (end <= 0) throw std::runtime_error("Artifact is empty: " + path.string());
    std::vector<char> bytes(static_cast<std::size_t>(end));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        throw std::runtime_error("Unable to read complete artifact: " + path.string());
    }
    return bytes;
}

}  // namespace

class TensorRTBackend::Impl {
public:
    ~Impl() { reset(); }

    Result<void> load(const BackendConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        reset();
        config_ = config;
        runtime_ = nvinfer1::createInferRuntime(logger_);
        if (!runtime_) return failure(ErrorCode::BackendFailure, "Unable to create TensorRT runtime");

        const bool engineExists = !config.enginePath.empty() && std::filesystem::is_regular_file(config.enginePath);
        if (engineExists) {
            auto loaded = deserialize(config.enginePath);
            if (loaded) {
                engine_ = loaded.value();
            } else if (config.loadPolicy == EngineLoadPolicy::EngineOnly ||
                       config.loadPolicy == EngineLoadPolicy::BuildIfMissing) {
                return loaded.status();
            }
        } else if (config.loadPolicy == EngineLoadPolicy::EngineOnly) {
            return failure(ErrorCode::ArtifactMissing, "TensorRT engine is required but does not exist",
                           config.enginePath.string());
        }

        if (!engine_) {
            if (config.onnxPath.empty() || !std::filesystem::is_regular_file(config.onnxPath)) {
                return failure(ErrorCode::ArtifactMissing, "ONNX model is required to build the TensorRT engine",
                               config.onnxPath.string());
            }
            auto built = build(config);
            if (!built) return built.status();
            engine_ = built.value();
            if (!config.enginePath.empty()) {
                auto saved = save(config.enginePath);
                if (!saved) return saved.status();
            }
        }

        context_ = engine_->createExecutionContext();
        if (!context_) return failure(ErrorCode::BackendFailure, "Unable to create TensorRT execution context");
        const auto streamError = cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking);
        if (streamError != cudaSuccess) return cudaStatus(streamError, "cudaStreamCreateWithFlags");

        try {
            collectSignature();
        } catch (const std::exception& error) {
            return failure(ErrorCode::TensorTypeMismatch, error.what());
        }
        loaded_ = true;
        return {};
    }

    Result<TensorMap> infer(const TensorMap& inputs) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_ || !context_) return failureResult<TensorMap>(ErrorCode::NotInitialized, "Backend is not loaded");

        for (const auto& spec : signature_.inputs) {
            const auto found = inputs.find(spec.name);
            if (found == inputs.end()) {
                return failureResult<TensorMap>(ErrorCode::InvalidArgument,
                                                "Required input tensor is missing", spec.name);
            }
            const Tensor& tensor = found->second;
            if (tensor.dtype != spec.dtype) {
                return failureResult<TensorMap>(ErrorCode::TensorTypeMismatch,
                                                "Input tensor data type does not match engine", spec.name);
            }
            const auto expectedBytes = tensor.shape.elementCount();
            if (!expectedBytes) return expectedBytes.status();
            if (expectedBytes.value() * dataTypeSize(tensor.dtype) != tensor.byteSize()) {
                return failureResult<TensorMap>(ErrorCode::TensorShapeMismatch,
                                                "Input byte size does not match its shape", spec.name);
            }
            auto dims = convertShape(tensor.shape);
            if (!dims) return dims.status();
            if (!context_->setInputShape(spec.name.c_str(), dims.value())) {
                return failureResult<TensorMap>(ErrorCode::TensorShapeMismatch,
                                                "TensorRT rejected the input shape",
                                                spec.name + " " + tensor.shape.toString());
            }
        }

        TensorMap outputs;
        for (const auto& spec : signature_.inputs) {
            const Tensor& tensor = inputs.at(spec.name);
            auto buffer = ensureBuffer(spec.name, tensor.byteSize());
            if (!buffer) return buffer.status();
            if (!context_->setTensorAddress(spec.name.c_str(), buffer.value())) {
                return failureResult<TensorMap>(ErrorCode::BackendFailure,
                                                "Unable to bind TensorRT input address", spec.name);
            }
        }

        for (const auto& spec : signature_.outputs) {
            const auto resolvedDims = context_->getTensorShape(spec.name.c_str());
            TensorShape resolvedShape = convertShape(resolvedDims);
            if (resolvedShape.isDynamic()) {
                return failureResult<TensorMap>(ErrorCode::TensorShapeMismatch,
                                                "TensorRT output shape remains dynamic after binding inputs",
                                                spec.name + " " + resolvedShape.toString());
            }
            auto elements = resolvedShape.elementCount();
            if (!elements) return elements.status();
            const std::size_t bytes = elements.value() * dataTypeSize(spec.dtype);
            auto buffer = ensureBuffer(spec.name, bytes);
            if (!buffer) return buffer.status();
            if (!context_->setTensorAddress(spec.name.c_str(), buffer.value())) {
                return failureResult<TensorMap>(ErrorCode::BackendFailure,
                                                "Unable to bind TensorRT output address", spec.name);
            }
            Tensor tensor;
            tensor.dtype = spec.dtype;
            tensor.shape = std::move(resolvedShape);
            tensor.bytes.resize(bytes);
            outputs.emplace(spec.name, std::move(tensor));
        }

        for (const auto& spec : signature_.inputs) {
            const Tensor& tensor = inputs.at(spec.name);
            const auto error = cudaMemcpyAsync(buffers_.at(spec.name).device, tensor.bytes.data(), tensor.byteSize(),
                                               cudaMemcpyHostToDevice, stream_);
            if (error != cudaSuccess) return cudaStatus(error, "cudaMemcpyAsync H2D for " + spec.name);
        }
        if (!context_->enqueueV3(stream_)) {
            return failureResult<TensorMap>(ErrorCode::BackendFailure, "TensorRT enqueueV3 failed");
        }
        for (auto& [name, tensor] : outputs) {
            const auto error = cudaMemcpyAsync(tensor.bytes.data(), buffers_.at(name).device, tensor.byteSize(),
                                               cudaMemcpyDeviceToHost, stream_);
            if (error != cudaSuccess) return cudaStatus(error, "cudaMemcpyAsync D2H for " + name);
        }
        const auto syncError = cudaStreamSynchronize(stream_);
        if (syncError != cudaSuccess) return cudaStatus(syncError, "cudaStreamSynchronize");
        return outputs;
    }

    const TensorSignature& signature() const noexcept { return signature_; }
    int maxBatchSize() const noexcept {
        if (!signature_.inputs.empty() && !signature_.inputs.front().shape.dims.empty() &&
            signature_.inputs.front().shape.dims.front() > 0) {
            return static_cast<int>(signature_.inputs.front().shape.dims.front());
        }
        return config_.maxBatchSize;
    }

private:
    struct DeviceBuffer {
        void* device{nullptr};
        std::size_t capacity{0};
    };

    template <typename T>
    Result<T> failureResult(ErrorCode code, std::string message, std::string context = {}) const {
        return Status::error(code, std::move(message), std::move(context));
    }

    Result<void> failure(ErrorCode code, std::string message, std::string context = {}) const {
        return Status::error(code, std::move(message), std::move(context));
    }

    Result<nvinfer1::ICudaEngine*> deserialize(const std::filesystem::path& path) {
        try {
            const auto bytes = readBinary(path);
            auto* engine = runtime_->deserializeCudaEngine(bytes.data(), bytes.size());
            if (!engine) {
                return failureResult<nvinfer1::ICudaEngine*>(
                    ErrorCode::EngineIncompatible,
                    "TensorRT could not deserialize the engine; version or GPU compatibility may differ",
                    path.string());
            }
            return engine;
        } catch (const std::exception& error) {
            return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::IoError, error.what(), path.string());
        }
    }

    Result<nvinfer1::ICudaEngine*> build(const BackendConfig& config) {
        auto* builder = nvinfer1::createInferBuilder(logger_);
        if (!builder) return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "Unable to create TensorRT builder");
        auto cleanupBuilder = [&] { delete builder; };
        auto* network = builder->createNetworkV2(0);
        if (!network) { cleanupBuilder(); return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "Unable to create TensorRT network"); }
        auto* parser = nvonnxparser::createParser(*network, logger_);
        if (!parser) { delete network; cleanupBuilder(); return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "Unable to create ONNX parser"); }
        if (!parser->parseFromFile(config.onnxPath.string().c_str(), static_cast<int>(nvinfer1::ILogger::Severity::kWARNING))) {
            std::ostringstream details;
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                if (i) details << " | ";
                details << parser->getError(i)->desc();
            }
            delete parser; delete network; cleanupBuilder();
            return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure,
                                                         "Unable to parse ONNX model", details.str());
        }
        auto* buildConfig = builder->createBuilderConfig();
        if (!buildConfig) { delete parser; delete network; cleanupBuilder(); return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "Unable to create TensorRT builder config"); }
        buildConfig->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, config.workspaceBytes);
        if (config.fp16 && builder->platformHasFastFp16()) buildConfig->setFlag(nvinfer1::BuilderFlag::kFP16);

        nvinfer1::IOptimizationProfile* profile = nullptr;
        for (int i = 0; i < network->getNbInputs(); ++i) {
            auto* input = network->getInput(i);
            auto dims = input->getDimensions();
            bool dynamic = false;
            for (int d = 0; d < dims.nbDims; ++d) dynamic = dynamic || dims.d[d] < 0;
            if (!dynamic) continue;
            if (!profile) profile = builder->createOptimizationProfile();
            auto minDims = dims;
            auto optDims = dims;
            auto maxDims = dims;
            for (int d = 0; d < dims.nbDims; ++d) {
                if (dims.d[d] >= 0) continue;
                int minValue = 1, optValue = 1, maxValue = 1;
                if (dims.nbDims == 4) {
                    if (d == 0) maxValue = std::max(1, config.maxBatchSize);
                    const bool nchw = config.inputLayout == TensorLayout::NCHW;
                    const int channelAxis = nchw ? 1 : 3;
                    const int heightAxis = nchw ? 2 : 1;
                    const int widthAxis = nchw ? 3 : 2;
                    if (d == channelAxis) minValue = optValue = maxValue = 3;
                    if (d == heightAxis) minValue = optValue = maxValue = config.inputHeight;
                    if (d == widthAxis) minValue = optValue = maxValue = config.inputWidth;
                }
                minDims.d[d] = minValue; optDims.d[d] = optValue; maxDims.d[d] = maxValue;
            }
            if (!profile ||
                !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN, minDims) ||
                !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT, optDims) ||
                !profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX, maxDims)) {
                delete buildConfig; delete parser; delete network; cleanupBuilder();
                return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure,
                                                             "Unable to configure TensorRT optimization profile",
                                                             input->getName());
            }
        }
        if (profile && (!profile->isValid() || buildConfig->addOptimizationProfile(profile) < 0)) {
            delete buildConfig; delete parser; delete network; cleanupBuilder();
            return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "TensorRT optimization profile is invalid");
        }

        auto* serialized = builder->buildSerializedNetwork(*network, *buildConfig);
        if (!serialized) {
            delete buildConfig; delete parser; delete network; cleanupBuilder();
            return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "TensorRT engine build failed");
        }
        auto* engine = runtime_->deserializeCudaEngine(serialized->data(), serialized->size());
        delete serialized; delete buildConfig; delete parser; delete network; cleanupBuilder();
        if (!engine) return failureResult<nvinfer1::ICudaEngine*>(ErrorCode::BackendFailure, "Unable to deserialize newly built TensorRT engine");
        return engine;
    }

    Result<void> save(const std::filesystem::path& path) {
        auto* serialized = engine_->serialize();
        if (!serialized) return failure(ErrorCode::BackendFailure, "Unable to serialize TensorRT engine");
        try {
            if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
            const auto temporary = path.string() + ".tmp";
            std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
            if (!stream || !stream.write(static_cast<const char*>(serialized->data()),
                                         static_cast<std::streamsize>(serialized->size()))) {
                delete serialized;
                return failure(ErrorCode::IoError, "Unable to write TensorRT engine", temporary);
            }
            stream.close();
            #ifdef _WIN32
            if (!MoveFileExW(std::filesystem::path(temporary).c_str(), path.c_str(),
                             MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH)) {
                const std::error_code errorCode(static_cast<int>(GetLastError()), std::system_category());
                delete serialized;
                return failure(ErrorCode::IoError, "Unable to atomically replace TensorRT engine cache",
                               errorCode.message());
            }
            #else
            std::filesystem::rename(temporary, path);
            #endif
            delete serialized;
            return {};
        } catch (const std::exception& error) {
            delete serialized;
            return failure(ErrorCode::IoError, error.what(), path.string());
        }
    }

    void collectSignature() {
        signature_ = {};
        const int count = engine_->getNbIOTensors();
        for (int i = 0; i < count; ++i) {
            const char* name = engine_->getIOTensorName(i);
            TensorSpec spec;
            spec.name = name;
            spec.dtype = convertType(engine_->getTensorDataType(name));
            spec.shape = convertShape(engine_->getTensorShape(name));
            spec.input = engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
            (spec.input ? signature_.inputs : signature_.outputs).push_back(std::move(spec));
        }
        if (signature_.inputs.empty() || signature_.outputs.empty()) {
            throw std::runtime_error("TensorRT engine must have at least one input and one output");
        }
    }

    Result<void*> ensureBuffer(const std::string& name, std::size_t requiredBytes) {
        auto& buffer = buffers_[name];
        if (buffer.capacity >= requiredBytes && buffer.device) return buffer.device;
        if (buffer.device) {
            cudaFree(buffer.device);
            buffer.device = nullptr;
            buffer.capacity = 0;
        }
        if (requiredBytes == 0) return failureResult<void*>(ErrorCode::TensorShapeMismatch, "Zero-sized tensors are not supported", name);
        const auto error = cudaMalloc(&buffer.device, requiredBytes);
        if (error != cudaSuccess) return cudaStatus(error, "cudaMalloc for " + name);
        buffer.capacity = requiredBytes;
        return buffer.device;
    }

    void reset() noexcept {
        loaded_ = false;
        for (auto& [_, buffer] : buffers_) if (buffer.device) cudaFree(buffer.device);
        buffers_.clear();
        if (stream_) { cudaStreamDestroy(stream_); stream_ = nullptr; }
        delete context_; context_ = nullptr;
        delete engine_; engine_ = nullptr;
        delete runtime_; runtime_ = nullptr;
        signature_ = {};
    }

    TensorRTLogger logger_;
    BackendConfig config_;
    TensorSignature signature_;
    nvinfer1::IRuntime* runtime_{nullptr};
    nvinfer1::ICudaEngine* engine_{nullptr};
    nvinfer1::IExecutionContext* context_{nullptr};
    cudaStream_t stream_{nullptr};
    std::unordered_map<std::string, DeviceBuffer> buffers_;
    std::mutex mutex_;
    bool loaded_{false};
};

TensorRTBackend::TensorRTBackend() : impl_(std::make_unique<Impl>()) {}
TensorRTBackend::~TensorRTBackend() = default;
TensorRTBackend::TensorRTBackend(TensorRTBackend&&) noexcept = default;
TensorRTBackend& TensorRTBackend::operator=(TensorRTBackend&&) noexcept = default;
Result<void> TensorRTBackend::load(const BackendConfig& config) { return impl_->load(config); }
Result<TensorMap> TensorRTBackend::infer(const TensorMap& inputs) { return impl_->infer(inputs); }
const TensorSignature& TensorRTBackend::signature() const noexcept { return impl_->signature(); }
int TensorRTBackend::maxBatchSize() const noexcept { return impl_->maxBatchSize(); }

}  // namespace anom::model
