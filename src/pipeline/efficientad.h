#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <unordered_map>

namespace anom::model {

struct EfficientADPipelineConfig {
    float hardWeight{1.0F};
    float softWeight{1.0F};
    float gaussianSigma{4.0F};
};

// Training-side EfficientAD pipeline. EfficientAD itself is distilled offline
// (teacher/student + autoencoder); this pipeline consumes the same three
// streams the runtime adapter uses, computes the combined anomaly map, and
// accumulates the score range observed over normal training images. The saved
// statistics are a reference for choosing the deployment postprocess
// normalization bounds and detection threshold.
class EfficientADPipeline final {
public:
    explicit EfficientADPipeline(EfficientADPipelineConfig config);

    Result<void> addBackendOutputs(
        const TensorMap& outputs,
        const std::unordered_map<std::string, std::string>& outputBindings,
        const EfficientADConfig& algorithmConfig,
        const cv::Size& inputSize);

    Result<void> addAnomalyMap(const cv::Mat& map);

    Result<void> saveStatistics(const std::filesystem::path& path) const;

    void clear();
    [[nodiscard]] std::size_t sampleCount() const noexcept { return sampleCount_; }

private:
    Result<void> validateConfig() const;
    void accumulate(const cv::Mat& combinedMap);

    EfficientADPipelineConfig config_;
    std::size_t sampleCount_{0};
    double imageMin_;
    double imageMax_;
    double pixelMin_;
    double pixelMax_;
};

}  // namespace anom::model
