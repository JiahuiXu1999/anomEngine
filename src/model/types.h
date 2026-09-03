#pragma once

#include "model/result.h"

#include <opencv2/core.hpp>

#include <chrono>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <unordered_map>
#include <vector>

namespace anom::model {

enum class DataType { Float32, Float16, Int32, Int64, UInt8, Bool };
enum class TensorLayout { NCHW, NHWC, NC, Unknown };
enum class AlgorithmType { PatchCore, Padim, Direct, EfficientAD, DFKDE, SPADE, Yolo };
enum class GraphContract { FeaturePyramid, Prediction };
enum class EngineLoadPolicy { EngineOnly, PreferEngine, BuildIfMissing };
enum class RuntimeBackend { TensorRT, OnnxRuntime };
enum class OrtExecutionProvider { Cpu, Cuda };
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

struct ImageGeometry {
    cv::Size originalSize;
    cv::Size resizedSize;
    cv::Rect cropRect;
};

struct PreprocessedBatch {
    Tensor tensor;
    std::vector<ImageGeometry> geometry;
};

// Algorithm-agnostic intermediate produced by every adapter. This is the single
// contract that lets any algorithm feed one shared postprocessing chain.
//   - `score`: image-level anomaly score in the algorithm's raw scale (NOT
//     normalized). Higher always means more anomalous.
//   - `anomalyMap`: per-pixel anomaly score map (CV_32FC1, floating point,
//     single channel, uncolored) in pre-processed (model-input) coordinates at
//     the adapter's native resolution. Higher always means more anomalous.
//     Empty when the algorithm produces no spatial output; the postprocessor
//     then degrades to image-level-only processing.
// Normalization, thresholding, geometry restoration, and analysis all happen
// downstream in AnomalyPostprocessor, never inside an adapter.
struct RawPrediction {
    float score{0.0F};
    cv::Mat anomalyMap;
};

using RawPredictionBatch = std::vector<RawPrediction>;

// A single connected anomalous region extracted from the binarized mask.
// All coordinates and areas are in restored (original image) space.
struct AnomalyRegion {
    cv::Rect boundingBox;      // Tight axis-aligned bounds, original image coordinates.
    cv::Point maxLocation{-1, -1};  // Pixel with the highest anomaly score in this region.
    double area{0.0};          // Pixel area of the region.
    float meanScore{0.0F};     // Mean anomaly score inside the region.
    float maxScore{0.0F};      // Peak anomaly score inside the region.
};

// Aggregated, algorithm-agnostic analysis derived from the anomaly map and mask.
struct AnomalyAnalysis {
    std::vector<AnomalyRegion> regions;  // Sorted by descending area.
    float anomalyAreaRatio{0.0F};  // Fraction of pixels flagged anomalous, in [0, 1].
    float meanScore{0.0F};         // Mean anomaly score over the anomalous region (0 if none).
    float maxScore{0.0F};          // Peak anomaly score over the whole map.
    int regionCount{0};            // Number of reported regions.
    bool hasMap{false};            // Whether a spatial map was available for analysis.
};

struct InferenceTiming {
    double preprocessMs{0.0};
    double backendMs{0.0};
    double adapterMs{0.0};
    double postprocessMs{0.0};
    double totalMs{0.0};
};

struct Prediction {
    float rawScore{0.0F};
    float score{0.0F};
    bool isAnomalous{false};
    cv::Mat rawAnomalyMap;  // Floating-point map restored to original image coordinates.
    cv::Mat anomalyMap;     // Normalized map (or raw map when normalization is disabled).
    cv::Mat mask;           // Binarized mask (empty when thresholding is disabled).
    AnomalyAnalysis analysis;
    std::string modelId;
    std::string modelVersion;
    InferenceTiming timing;
};

using PredictionBatch = std::vector<Prediction>;

struct ModelInfo {
    std::string id;
    std::string version;
    AlgorithmType algorithm{AlgorithmType::Direct};
    GraphContract graphContract{GraphContract::Prediction};
    RuntimeBackend runtimeBackend{RuntimeBackend::TensorRT};
    std::string executionProvider{"cuda"};
    TensorSignature signature;
};

struct ExecutionInfo {
    DevicePreference requestedDevice{DevicePreference::Manifest};
    RuntimeBackend runtimeBackend{RuntimeBackend::TensorRT};
    OrtExecutionProvider executionProvider{OrtExecutionProvider::Cuda};
    int deviceId{0};
    std::string deviceName;
    PrecisionPreference precision{PrecisionPreference::Auto};
    bool fallbackOccurred{false};
    std::string fallbackReason;
};

[[nodiscard]] std::size_t dataTypeSize(DataType type);
[[nodiscard]] const char* toString(DataType type) noexcept;
[[nodiscard]] const char* toString(AlgorithmType type) noexcept;
[[nodiscard]] const char* toString(RuntimeBackend type) noexcept;
[[nodiscard]] const char* toString(OrtExecutionProvider type) noexcept;

}  // namespace anom::model
