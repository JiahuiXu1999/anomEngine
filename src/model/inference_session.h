#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <memory>
#include <optional>

namespace anom::model {

class AnomalyPostprocessor;
class ImagePreprocessor;
class IModelAdapter;
class IRuntimeBackend;

struct LoadOptions {
    std::optional<RuntimeBackend> backend;
    std::optional<EngineLoadPolicy> engineLoadPolicy;
    std::optional<bool> fp16;
    std::optional<int> deviceId;
    DevicePreference devicePreference{DevicePreference::Manifest};
    FallbackPolicy fallbackPolicy{FallbackPolicy::None};
    PrecisionPreference precision{PrecisionPreference::Manifest};
    std::filesystem::path pluginDirectory;
    bool warmup{false};
};

class InferenceSession {
public:
    static Result<std::unique_ptr<InferenceSession>> load(
        const std::filesystem::path& modelPackage,
        const LoadOptions& options = {});

    ~InferenceSession();
    InferenceSession(const InferenceSession&) = delete;
    InferenceSession& operator=(const InferenceSession&) = delete;

    Result<Prediction> predict(const cv::Mat& image);
    Result<PredictionBatch> predictBatch(const std::vector<cv::Mat>& images);
    Result<void> warmup();

    [[nodiscard]] const ModelInfo& modelInfo() const noexcept { return modelInfo_; }
    [[nodiscard]] const ExecutionInfo& executionInfo() const noexcept { return executionInfo_; }

private:
    InferenceSession(ModelPackage package,
                     std::unique_ptr<IRuntimeBackend> backend,
                     std::unique_ptr<IModelAdapter> adapter);

    Result<PredictionBatch> predictChunk(const std::vector<cv::Mat>& images);

    ModelPackage package_;
    std::unique_ptr<IRuntimeBackend> backend_;
    std::unique_ptr<IModelAdapter> adapter_;
    std::unique_ptr<ImagePreprocessor> preprocessor_;
    std::unique_ptr<AnomalyPostprocessor> postprocessor_;
    ModelInfo modelInfo_;
    ExecutionInfo executionInfo_;
};

}  // namespace anom::model
