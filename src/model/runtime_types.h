#pragma once

#include "model/result.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace anom::model {

enum class DataType { Float32, Float16, Int32, Int64, UInt8, Bool };
enum class TensorLayout { NCHW, NHWC, NC, Unknown };
enum class AlgorithmType { PatchCore, Padim, Direct, EfficientAD, DFKDE, SPADE, Yolo };
enum class GraphContract { FeaturePyramid, Prediction };
enum class EngineLoadPolicy { EngineOnly, PreferEngine, BuildIfMissing };
enum class RuntimeBackend { TensorRT, OnnxRuntime };
// Values also form the optional backend execution extension's C ABI.
enum class ExecutionProvider { Default = 0, Cpu = 1, Cuda = 2 };
enum class OrtGraphOptimization { Disabled, Basic, Extended, All };
enum class OrtExecutionMode { Sequential, Parallel };
enum class DevicePreference { Manifest, Auto, Cpu, Gpu };
enum class FallbackPolicy { None, LoadOnly };
enum class PrecisionPreference { Manifest, Auto, Float32, Float16 };

struct TensorShape {
    std::vector<std::int64_t> dims;

    [[nodiscard]] bool isDynamic() const noexcept;
    [[nodiscard]] Result<std::size_t> elementCount() const;
    [[nodiscard]] std::string toString() const;
};

struct Tensor {
    DataType dtype{DataType::Float32};
    TensorShape shape;
    std::vector<std::byte> bytes;

    // Plugin adapters/backends may consume a read-only view owned by another
    // module. Regular tensors continue to store their bytes in `bytes`.
    const std::byte* externalData{nullptr};
    std::size_t externalByteSize{0};
    std::shared_ptr<void> externalOwner;

    template <typename T>
    T* data() {
        // Mutable access materializes a private copy; const readers keep the
        // shared plugin output without copying or modifying another consumer.
        if (externalData) {
            std::vector<std::byte> owned(externalByteSize);
            if (externalByteSize) std::memcpy(owned.data(), externalData, externalByteSize);
            bytes = std::move(owned);
            externalData = nullptr;
            externalByteSize = 0;
            externalOwner.reset();
        }
        return reinterpret_cast<T*>(bytes.data());
    }

    template <typename T>
    const T* data() const {
        return reinterpret_cast<const T*>(externalData ? externalData : bytes.data());
    }

    void setExternalView(const void* data, std::size_t size,
                         std::shared_ptr<void> owner = {}) noexcept {
        externalData = static_cast<const std::byte*>(data);
        externalByteSize = size;
        externalOwner = std::move(owner);
    }

    [[nodiscard]] std::size_t byteSize() const noexcept {
        return externalData ? externalByteSize : bytes.size();
    }
};

using TensorMap = std::unordered_map<std::string, Tensor>;

struct TensorSpec {
    std::string name;
    DataType dtype{DataType::Float32};
    TensorShape shape;
    bool input{false};
};

struct TensorSignature {
    std::vector<TensorSpec> inputs;
    std::vector<TensorSpec> outputs;

    [[nodiscard]] const TensorSpec* findInput(const std::string& name) const;
    [[nodiscard]] const TensorSpec* findOutput(const std::string& name) const;
};

[[nodiscard]] std::size_t dataTypeSize(DataType type);
[[nodiscard]] const char* toString(DataType type) noexcept;
[[nodiscard]] const char* toString(AlgorithmType type) noexcept;
[[nodiscard]] const char* toString(RuntimeBackend type) noexcept;

}  // namespace anom::model
