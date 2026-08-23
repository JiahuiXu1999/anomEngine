#include "backends/ort_backend.h"

#include <onnxruntime_cxx_api.h>
#include <onnxruntime_session_options_config_keys.h>

#if defined(ANOM_ORT_ENABLE_CUDA)
#include <cuda_provider_factory.h>
#endif

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <limits>
#include <mutex>
#include <new>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace anom::model {
namespace {

DataType fromOrtType(ONNXTensorElementDataType type) {
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: return DataType::Float32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return DataType::Float16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: return DataType::Int32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: return DataType::Int64;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: return DataType::UInt8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: return DataType::Bool;
        default: throw std::runtime_error("Unsupported ONNX tensor element type " +
                                          std::to_string(static_cast<int>(type)));
    }
}

ONNXTensorElementDataType toOrtType(DataType type) {
    switch (type) {
        case DataType::Float32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case DataType::Float16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case DataType::Int32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case DataType::Int64: return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        case DataType::UInt8: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case DataType::Bool: return ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL;
    }
    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

GraphOptimizationLevel graphOptimizationLevel(OrtGraphOptimization level) {
    switch (level) {
        case OrtGraphOptimization::Disabled: return ORT_DISABLE_ALL;
        case OrtGraphOptimization::Basic: return ORT_ENABLE_BASIC;
        case OrtGraphOptimization::Extended: return ORT_ENABLE_EXTENDED;
        case OrtGraphOptimization::All: return ORT_ENABLE_ALL;
    }
    return ORT_ENABLE_ALL;
}

bool compatibleShape(const TensorShape& modelShape, const TensorShape& actualShape) {
    if (modelShape.dims.size() != actualShape.dims.size()) return false;
    for (std::size_t i = 0; i < modelShape.dims.size(); ++i) {
        if (actualShape.dims[i] <= 0) return false;
        if (modelShape.dims[i] >= 0 && modelShape.dims[i] != actualShape.dims[i]) return false;
    }
    return true;
}

Status ortFailure(const char* operation, const Ort::Exception& error) {
    ErrorCode code = ErrorCode::BackendFailure;
    if (error.GetOrtErrorCode() == ORT_INVALID_ARGUMENT) code = ErrorCode::InvalidArgument;
    if (error.GetOrtErrorCode() == ORT_NO_SUCHFILE) code = ErrorCode::ArtifactMissing;
    return Status::error(code, std::string(operation) + " failed", error.what());
}

}  // namespace

class OrtBackend::Impl {
public:
    Impl() : env_(ORT_LOGGING_LEVEL_WARNING, "anomEngine") {}

    Result<void> load(const BackendConfig& config) {
        std::lock_guard<std::mutex> lock(mutex_);
        session_.reset();
        signature_ = {};
        loaded_ = false;
        config_ = config;

        if (config.onnxPath.empty() || !std::filesystem::is_regular_file(config.onnxPath)) {
            return Status::error(ErrorCode::ArtifactMissing,
                                 "ONNX model does not exist", config.onnxPath.string());
        }

        try {
            Ort::SessionOptions options;
            options.SetGraphOptimizationLevel(graphOptimizationLevel(config.ortGraphOptimization));
            options.SetExecutionMode(config.ortExecutionMode == OrtExecutionMode::Parallel
                                         ? ORT_PARALLEL
                                         : ORT_SEQUENTIAL);
            if (config.intraOpThreads > 0) options.SetIntraOpNumThreads(config.intraOpThreads);
            if (config.interOpThreads > 0) options.SetInterOpNumThreads(config.interOpThreads);
            if (config.enableMemoryPattern) options.EnableMemPattern();
            else options.DisableMemPattern();
            if (config.enableCpuMemoryArena) options.EnableCpuMemArena();
            else options.DisableCpuMemArena();

            if (config.enableProfiling) {
#if defined(_WIN32)
                const std::wstring prefix = config.profileFilePrefix.wstring();
#else
                const std::string prefix = config.profileFilePrefix.string();
#endif
                options.EnableProfiling(prefix.c_str());
            }

            if (config.ortProvider == OrtExecutionProvider::Cuda) {
#if defined(ANOM_ORT_ENABLE_CUDA)
                Ort::ThrowOnError(
                    OrtSessionOptionsAppendExecutionProvider_CUDA(options, config.deviceId));
                if (config.ortStrictProvider) {
                    // ORT otherwise assigns unsupported CUDA nodes to the default CPU EP.
                    options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1");
                }
#else
                return Status::error(
                    ErrorCode::BackendFailure,
                    "CUDAExecutionProvider was requested but this build uses CPU-only ONNX Runtime",
                    "Configure with -DANOM_ORT_ENABLE_CUDA=ON and a GPU ONNX Runtime package");
#endif
            }

            session_ = std::make_unique<Ort::Session>(env_, config.onnxPath.c_str(), options);
            collectSignature();
            loaded_ = true;
            return {};
        } catch (const Ort::Exception& error) {
            session_.reset();
            return ortFailure("ONNX Runtime session creation", error);
        } catch (const std::bad_alloc&) {
            session_.reset();
            return Status::error(ErrorCode::OutOfMemory,
                                 "Out of memory while loading ONNX Runtime model");
        } catch (const std::exception& error) {
            session_.reset();
            return Status::error(ErrorCode::TensorTypeMismatch,
                                 "Unable to inspect ONNX model signature", error.what());
        }
    }

    Result<TensorMap> infer(const TensorMap& inputs) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!loaded_ || !session_) {
            return Status::error(ErrorCode::NotInitialized,
                                 "ONNX Runtime backend is not loaded");
        }
        if (inputs.size() != signature_.inputs.size()) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "Input tensor count does not match the ONNX model",
                                 std::to_string(inputs.size()) + " provided, " +
                                     std::to_string(signature_.inputs.size()) + " required");
        }

        try {
            Ort::MemoryInfo memoryInfo =
                Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
            std::vector<const char*> inputNames;
            std::vector<Ort::Value> inputValues;
            inputNames.reserve(signature_.inputs.size());
            inputValues.reserve(signature_.inputs.size());

            for (const auto& spec : signature_.inputs) {
                const auto found = inputs.find(spec.name);
                if (found == inputs.end()) {
                    return Status::error(ErrorCode::InvalidArgument,
                                         "Required ONNX input tensor is missing", spec.name);
                }
                const Tensor& tensor = found->second;
                if (tensor.dtype != spec.dtype) {
                    return Status::error(ErrorCode::TensorTypeMismatch,
                                         "Input tensor data type does not match ONNX model",
                                         spec.name + ": expected " + toString(spec.dtype) +
                                             ", got " + toString(tensor.dtype));
                }
                if (!compatibleShape(spec.shape, tensor.shape)) {
                    return Status::error(ErrorCode::TensorShapeMismatch,
                                         "Input tensor shape does not match ONNX model",
                                         spec.name + ": expected " + spec.shape.toString() +
                                             ", got " + tensor.shape.toString());
                }
                auto elements = tensor.shape.elementCount();
                if (!elements) return elements.status();
                if (elements.value() >
                    std::numeric_limits<std::size_t>::max() / dataTypeSize(tensor.dtype) ||
                    elements.value() * dataTypeSize(tensor.dtype) != tensor.byteSize()) {
                    return Status::error(ErrorCode::TensorShapeMismatch,
                                         "Input tensor byte size does not match its shape", spec.name);
                }

                inputNames.push_back(spec.name.c_str());
                inputValues.emplace_back(Ort::Value::CreateTensor(
                    memoryInfo, const_cast<std::byte*>(tensor.bytes.data()), tensor.byteSize(),
                    tensor.shape.dims.data(), tensor.shape.dims.size(), toOrtType(tensor.dtype)));
            }

            std::vector<const char*> outputNames;
            outputNames.reserve(signature_.outputs.size());
            for (const auto& spec : signature_.outputs) outputNames.push_back(spec.name.c_str());

            Ort::RunOptions runOptions;
            auto ortOutputs = session_->Run(runOptions, inputNames.data(), inputValues.data(),
                                            inputValues.size(), outputNames.data(), outputNames.size());
            if (ortOutputs.size() != signature_.outputs.size()) {
                return Status::error(ErrorCode::BackendFailure,
                                     "ONNX Runtime returned an unexpected output count");
            }

            TensorMap outputs;
            outputs.reserve(ortOutputs.size());
            for (std::size_t i = 0; i < ortOutputs.size(); ++i) {
                Ort::Value& value = ortOutputs[i];
                const TensorSpec& spec = signature_.outputs[i];
                if (!value.IsTensor()) {
                    return Status::error(ErrorCode::TensorTypeMismatch,
                                         "ONNX output is not a dense tensor", spec.name);
                }
                const auto info = value.GetTensorTypeAndShapeInfo();
                const DataType actualType = fromOrtType(info.GetElementType());
                if (actualType != spec.dtype) {
                    return Status::error(ErrorCode::TensorTypeMismatch,
                                         "ONNX Runtime output type differs from model signature", spec.name);
                }

                Tensor tensor;
                tensor.dtype = actualType;
                tensor.shape.dims = info.GetShape();
                auto elements = tensor.shape.elementCount();
                if (!elements) return elements.status();
                if (elements.value() >
                    std::numeric_limits<std::size_t>::max() / dataTypeSize(tensor.dtype)) {
                    return Status::error(ErrorCode::TensorShapeMismatch,
                                         "ONNX output byte size overflows size_t", spec.name);
                }
                tensor.bytes.resize(elements.value() * dataTypeSize(tensor.dtype));
                if (!tensor.bytes.empty()) {
                    std::memcpy(tensor.bytes.data(), value.GetTensorRawData(), tensor.bytes.size());
                }
                outputs.emplace(spec.name, std::move(tensor));
            }
            return outputs;
        } catch (const Ort::Exception& error) {
            return ortFailure("ONNX Runtime inference", error);
        } catch (const std::bad_alloc&) {
            return Status::error(ErrorCode::OutOfMemory,
                                 "Out of memory during ONNX Runtime inference");
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::BackendFailure,
                                 "Unexpected ONNX Runtime inference failure", error.what());
        }
    }

    const TensorSignature& signature() const noexcept { return signature_; }

    int maxBatchSize() const noexcept {
        if (!signature_.inputs.empty() && !signature_.inputs.front().shape.dims.empty() &&
            signature_.inputs.front().shape.dims.front() > 0) {
            const auto batch = signature_.inputs.front().shape.dims.front();
            return batch > std::numeric_limits<int>::max() ? 1 : static_cast<int>(batch);
        }
        return std::max(1, config_.maxBatchSize);
    }

private:
    TensorSpec tensorSpec(std::size_t index, bool input, Ort::AllocatorWithDefaultOptions& allocator) {
        Ort::AllocatedStringPtr name = input
            ? session_->GetInputNameAllocated(index, allocator)
            : session_->GetOutputNameAllocated(index, allocator);
        Ort::TypeInfo typeInfo = input ? session_->GetInputTypeInfo(index)
                                       : session_->GetOutputTypeInfo(index);
        if (typeInfo.GetONNXType() != ONNX_TYPE_TENSOR) {
            throw std::runtime_error(std::string(input ? "Input '" : "Output '") + name.get() +
                                     "' is not a dense tensor");
        }
        auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
        TensorSpec spec;
        spec.name = name.get();
        spec.input = input;
        spec.dtype = fromOrtType(tensorInfo.GetElementType());
        spec.shape.dims = tensorInfo.GetShape();
        return spec;
    }

    void collectSignature() {
        Ort::AllocatorWithDefaultOptions allocator;
        const std::size_t inputCount = session_->GetInputCount();
        const std::size_t outputCount = session_->GetOutputCount();
        if (inputCount == 0 || outputCount == 0) {
            throw std::runtime_error("ONNX model must expose at least one input and one output");
        }
        signature_.inputs.reserve(inputCount);
        signature_.outputs.reserve(outputCount);
        for (std::size_t i = 0; i < inputCount; ++i) {
            signature_.inputs.push_back(tensorSpec(i, true, allocator));
        }
        for (std::size_t i = 0; i < outputCount; ++i) {
            signature_.outputs.push_back(tensorSpec(i, false, allocator));
        }
    }

    Ort::Env env_;
    std::unique_ptr<Ort::Session> session_;
    BackendConfig config_;
    TensorSignature signature_;
    bool loaded_{false};
    mutable std::mutex mutex_;
};

OrtBackend::OrtBackend() : impl_(std::make_unique<Impl>()) {}
OrtBackend::~OrtBackend() = default;
OrtBackend::OrtBackend(OrtBackend&&) noexcept = default;
OrtBackend& OrtBackend::operator=(OrtBackend&&) noexcept = default;
Result<void> OrtBackend::load(const BackendConfig& config) { return impl_->load(config); }
Result<TensorMap> OrtBackend::infer(const TensorMap& inputs) { return impl_->infer(inputs); }
const TensorSignature& OrtBackend::signature() const noexcept { return impl_->signature(); }
int OrtBackend::maxBatchSize() const noexcept { return impl_->maxBatchSize(); }

}  // namespace anom::model
