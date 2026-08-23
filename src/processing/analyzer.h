#pragma once

#include "model/result.h"
#include "model/types.h"

#include <opencv2/core.hpp>

namespace anom::model {

struct AnalysisConfig {
    int minRegionArea{16};  // Minimum pixel area for a region to be reported.
    int maxRegions{0};      // Report at most this many regions (0 = unlimited).
};

// Algorithm-agnostic connected-component analyzer. It consumes a normalized
// anomaly map and a binarized mask (both in original image coordinates) and
// produces AnomalyAnalysis: per-region bounds, area, and score statistics plus
// image-level aggregates.
class AnomalyAnalyzer {
public:
    explicit AnomalyAnalyzer(AnalysisConfig config) : config_(config) {}

    Result<AnomalyAnalysis> analyze(const cv::Mat& anomalyMap, const cv::Mat& mask) const;

private:
    AnalysisConfig config_;
};

}  // namespace anom::model
