#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"
#include "processing/analyzer.h"

namespace anom::model {

// Unified, algorithm-agnostic postprocessing chain. Every adapter produces a
// RawPredictionBatch (raw score + anomaly map); this class turns it into a
// PredictionBatch by running a configurable stage chain in a fixed order:
//
//   geometry restore -> normalize -> smooth -> threshold -> morphology -> analyze
//
// The adapter never performs these steps, so behavior stays consistent across
// PatchCore, PaDiM, EfficientAD, and Direct models.
class AnomalyPostprocessor {
public:
    AnomalyPostprocessor(PostprocessConfig config, std::string modelId, std::string modelVersion)
        : config_(std::move(config)), modelId_(std::move(modelId)), modelVersion_(std::move(modelVersion)),
          analyzer_(AnalysisConfig{config_.minRegionArea, config_.maxRegions}) {}

    Result<PredictionBatch> process(
        const RawPredictionBatch& raw,
        const std::vector<ImageGeometry>& geometry) const;

private:
    [[nodiscard]] float normalize(float value, float threshold, float minimum, float maximum) const;
    [[nodiscard]] cv::Mat restoreGeometry(const cv::Mat& map, const ImageGeometry& geometry) const;
    [[nodiscard]] Result<void> smooth(cv::Mat& map) const;
    [[nodiscard]] Result<void> morph(cv::Mat& mask) const;

    PostprocessConfig config_;
    std::string modelId_;
    std::string modelVersion_;
    AnomalyAnalyzer analyzer_;
};

}  // namespace anom::model
