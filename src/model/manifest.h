#pragma once

#include "model/result.h"
#include "model/types.h"

#include <opencv2/imgproc.hpp>

#include <array>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace anom::model {

struct InputConfig {
    std::string tensorName{"input"};
    TensorLayout layout{TensorLayout::NCHW};
    std::string color{"RGB"};
    int resizeHeight{224};
    int resizeWidth{224};
    std::optional<cv::Size> centerCrop;
    int interpolation{cv::INTER_LINEAR};
    std::array<float, 3> mean{0.485F, 0.456F, 0.406F};
    std::array<float, 3> std{0.229F, 0.224F, 0.225F};
};

struct RuntimeConfig {
    RuntimeBackend backend{RuntimeBackend::TensorRT};
    std::filesystem::path onnxFile;
    std::filesystem::path engineFile;
    EngineLoadPolicy loadPolicy{EngineLoadPolicy::PreferEngine};
    bool fp16{false};
    int maxBatchSize{1};
    std::size_t workspaceBytes{2ULL * 1024ULL * 1024ULL * 1024ULL};

    OrtGraphOptimization ortGraphOptimization{OrtGraphOptimization::All};
    OrtExecutionMode ortExecutionMode{OrtExecutionMode::Sequential};
    int deviceId{0};
    int intraOpThreads{0};
    int interOpThreads{0};
    bool enableMemoryPattern{true};
    bool enableCpuMemoryArena{true};
    bool enableProfiling{false};
    std::filesystem::path profileFilePrefix{"anom_ort_profile"};
};

struct PostprocessConfig {
    bool normalize{true};
    bool threshold{true};
    float imageThreshold{0.5F};
    float pixelThreshold{0.5F};
    float imageMin{0.0F};
    float imageMax{1.0F};
    float pixelMin{0.0F};
    float pixelMax{1.0F};
    float imageSensitivity{0.5F};
    float pixelSensitivity{0.5F};

    // Optional smoothing stage applied to the anomaly map before thresholding.
    bool smooth{false};
    int smoothKernel{3};      // Must be a positive odd integer.
    float smoothSigma{0.0F};  // <= 0 derives sigma from the kernel size.

    // Optional morphological cleanup applied to the mask after thresholding.
    bool morphology{false};
    int morphologyKernel{5};       // Must be a positive odd integer.
    int morphologyIterations{1};
    std::string morphologyMode{"open"};  // "open" | "close"

    // Optional connected-component analysis producing AnomalyAnalysis.
    bool analyze{false};
    int minRegionArea{16};  // Minimum pixel area for a region to be reported.
    int maxRegions{0};      // Report at most this many regions (0 = unlimited).
};

struct PatchCoreConfig {
    std::filesystem::path indexFile;
    std::vector<std::string> featureLayers;
    int embeddingDimension{0};
    int numNeighbors{1};
    int poolingKernel{3};
    int poolingStride{1};
    int poolingPadding{1};
    float gaussianSigma{4.0F};
    bool sqrtDistances{true};
    bool weightedImageScore{true};
};

struct PadimConfig {
    std::filesystem::path statisticsFile;
    std::filesystem::path channelIndicesFile;
    std::vector<std::string> featureLayers;
    int embeddingDimension{0};
    int featureHeight{0};
    int featureWidth{0};
    float gaussianSigma{4.0F};
};

struct DirectConfig {
    std::string scoreSemantic{"pred_score"};
    std::string mapSemantic{"anomaly_map"};
};

// EfficientAD deploys as an end-to-end teacher/student + autoencoder model.
// The backend returns a teacher feature stream, a student feature stream, and a
// reconstruction-error map (computed inside the graph from the autoencoder and
// its input). The adapter derives the hard score from the teacher/student
// discrepancy and combines it with the soft reconstruction error.
struct EfficientADConfig {
    std::string teacherSemantic{"teacher"};
    std::string studentSemantic{"student"};
    std::string reconstructionErrorSemantic{"reconstruction_error"};
    float hardWeight{1.0F};
    float softWeight{1.0F};
    float gaussianSigma{4.0F};
};

// DFKDE models the normal feature distribution with kernel density estimation.
// Training reduces the aggregated pyramid embedding with PCA and stores the
// projected gallery; inference scores each patch by the negative log density.
struct DFKDEConfig {
    std::filesystem::path statisticsFile;
    std::vector<std::string> featureLayers;
    int embeddingDimension{0};
    int nComponents{16};
    float kernelSigma{10.0F};
    float gaussianSigma{4.0F};
};

// SPADE performs per-layer k-nearest-neighbor retrieval on a feature pyramid
// and fuses the resulting multi-resolution anomaly maps. Each feature layer has
// its own FAISS gallery, so indexFiles aligns one-to-one with featureLayers.
struct SPADEConfig {
    std::vector<std::string> featureLayers;
    std::vector<std::string> indexFiles;
    int numNeighbors{5};
    float gaussianSigma{4.0F};
};

// YOLO deploys as a supervised object detector. The exported graph returns the
// raw anchor-free detection head (e.g. YOLOv8 [N, 4+C, M]): the first four rows
// are already-decoded boxes (cx, cy, w, h in absolute input pixels) and the
// remaining C rows are per-class sigmoid scores. The adapter filters by
// confidence, runs non-maximum suppression, and maps the surviving boxes to an
// anomaly prediction — a detected box means "anomalous", with the highest
// confidence as the image score and the box regions painted into the anomaly map.
struct YoloConfig {
    std::string detectionSemantic{"output"};
    int numClasses{1};
    float confThreshold{0.25F};
    float nmsThreshold{0.45F};
};

using AlgorithmConfig =
    std::variant<PatchCoreConfig, PadimConfig, DirectConfig, EfficientADConfig,
                 DFKDEConfig, SPADEConfig, YoloConfig>;

struct ModelManifest {
    int schemaVersion{1};
    std::string modelId;
    std::string modelVersion;
    AlgorithmType algorithm{AlgorithmType::Direct};
    GraphContract graphContract{GraphContract::Prediction};
    InputConfig input;
    RuntimeConfig runtime;
    PostprocessConfig postprocess;
    std::unordered_map<std::string, std::string> outputs;  // semantic -> tensor name
    std::unordered_map<std::string, std::string> checksums;  // relative artifact path -> lowercase SHA-256
    AlgorithmConfig algorithmConfig{DirectConfig{}};
};

class ModelPackage {
public:
    static Result<ModelPackage> load(const std::filesystem::path& packagePath);

    [[nodiscard]] const ModelManifest& manifest() const noexcept { return manifest_; }
    [[nodiscard]] const std::filesystem::path& root() const noexcept { return root_; }
    [[nodiscard]] const std::filesystem::path& manifestPath() const noexcept { return manifestPath_; }

    Result<std::filesystem::path> resolveArtifact(
        const std::filesystem::path& relativePath,
        bool mustExist = true) const;

private:
    std::filesystem::path root_;
    std::filesystem::path manifestPath_;
    ModelManifest manifest_;
};

}  // namespace anom::model
