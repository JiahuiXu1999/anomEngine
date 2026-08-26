#include "model/types.h"

#include <limits>
#include <sstream>

namespace anom::model {

bool TensorShape::isDynamic() const noexcept {
    for (const auto dim : dims) {
        if (dim < 0) return true;
    }
    return false;
}

Result<std::size_t> TensorShape::elementCount() const {
    std::size_t count = 1;
    for (const auto dim : dims) {
        if (dim < 0) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Cannot compute element count for a dynamic shape", toString());
        }
        if (dim == 0) return static_cast<std::size_t>(0);
        const auto value = static_cast<std::size_t>(dim);
        if (count > std::numeric_limits<std::size_t>::max() / value) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Tensor element count overflows size_t", toString());
        }
        count *= value;
    }
    return count;
}

std::string TensorShape::toString() const {
    std::ostringstream stream;
    stream << '[';
    for (std::size_t i = 0; i < dims.size(); ++i) {
        if (i) stream << 'x';
        stream << dims[i];
    }
    stream << ']';
    return stream.str();
}

const TensorSpec* TensorSignature::findInput(const std::string& name) const {
    for (const auto& item : inputs) if (item.name == name) return &item;
    return nullptr;
}

const TensorSpec* TensorSignature::findOutput(const std::string& name) const {
    for (const auto& item : outputs) if (item.name == name) return &item;
    return nullptr;
}

std::size_t dataTypeSize(DataType type) {
    switch (type) {
        case DataType::Float32: return 4;
        case DataType::Float16: return 2;
        case DataType::Int32: return 4;
        case DataType::Int64: return 8;
        case DataType::UInt8: return 1;
        case DataType::Bool: return 1;
    }
    return 0;
}

const char* toString(DataType type) noexcept {
    switch (type) {
        case DataType::Float32: return "float32";
        case DataType::Float16: return "float16";
        case DataType::Int32: return "int32";
        case DataType::Int64: return "int64";
        case DataType::UInt8: return "uint8";
        case DataType::Bool: return "bool";
    }
    return "unknown";
}

const char* toString(AlgorithmType type) noexcept {
    switch (type) {
        case AlgorithmType::PatchCore: return "patchcore";
        case AlgorithmType::Padim: return "padim";
        case AlgorithmType::Direct: return "direct";
        case AlgorithmType::EfficientAD: return "efficientad";
        case AlgorithmType::DFKDE: return "dfkde";
        case AlgorithmType::SPADE: return "spade";
        case AlgorithmType::Yolo: return "yolo";
    }
    return "unknown";
}

const char* toString(RuntimeBackend type) noexcept {
    switch (type) {
        case RuntimeBackend::TensorRT: return "tensorrt";
        case RuntimeBackend::OnnxRuntime: return "onnxruntime";
    }
    return "unknown";
}

const char* toString(OrtExecutionProvider type) noexcept {
    switch (type) {
        case OrtExecutionProvider::Cpu: return "cpu";
        case OrtExecutionProvider::Cuda: return "cuda";
    }
    return "unknown";
}

}  // namespace anom::model
