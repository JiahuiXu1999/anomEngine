#include "processing/analyzer.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>

namespace anom::model {

Result<AnomalyAnalysis> AnomalyAnalyzer::analyze(const cv::Mat& anomalyMap, const cv::Mat& mask) const {
    AnomalyAnalysis analysis;
    analysis.hasMap = !anomalyMap.empty();

    if (anomalyMap.empty() && mask.empty()) return analysis;

    // Whole-map peak score, available even without a mask.
    if (!anomalyMap.empty()) {
        double minimum = 0.0;
        double maximum = 0.0;
        cv::minMaxLoc(anomalyMap, &minimum, &maximum);
        analysis.maxScore = static_cast<float>(maximum);
    }

    if (mask.empty() || mask.type() != CV_8UC1) return analysis;

    const double total = static_cast<double>(mask.total());
    const double foreground = static_cast<double>(cv::countNonZero(mask));
    if (total > 0.0) analysis.anomalyAreaRatio = static_cast<float>(foreground / total);

    cv::Mat labels;
    const int componentCount = cv::connectedComponents(mask, labels, 8, CV_32S);
    if (componentCount <= 1) return analysis;  // Only background.

    std::vector<AnomalyRegion> regions;
    regions.reserve(static_cast<std::size_t>(componentCount - 1));

    for (int label = 1; label < componentCount; ++label) {
        cv::Mat regionMask;
        cv::compare(labels, label, regionMask, cv::CMP_EQ);
        const double area = static_cast<double>(cv::countNonZero(regionMask));
        if (area < static_cast<double>(config_.minRegionArea)) continue;

        std::vector<cv::Point> pixels;
        cv::findNonZero(regionMask, pixels);
        if (pixels.empty()) continue;

        AnomalyRegion region;
        region.boundingBox = cv::boundingRect(pixels);
        region.area = area;

        if (!anomalyMap.empty()) {
            double minimum = 0.0;
            double maximum = 0.0;
            cv::Point minLocation;
            cv::Point maxLocation;
            cv::minMaxLoc(anomalyMap, &minimum, &maximum, &minLocation, &maxLocation, regionMask);
            region.maxScore = static_cast<float>(maximum);
            region.maxLocation = maxLocation;
            region.meanScore = static_cast<float>(cv::mean(anomalyMap, regionMask)[0]);
        }
        regions.push_back(region);
    }

    std::sort(regions.begin(), regions.end(),
              [](const AnomalyRegion& lhs, const AnomalyRegion& rhs) {
                  return lhs.area > rhs.area;
              });

    if (config_.maxRegions > 0 && regions.size() > static_cast<std::size_t>(config_.maxRegions)) {
        regions.resize(static_cast<std::size_t>(config_.maxRegions));
    }

    analysis.regionCount = static_cast<int>(regions.size());
    if (!regions.empty()) {
        double sum = 0.0;
        for (const auto& region : regions) sum += region.meanScore;
        analysis.meanScore = static_cast<float>(sum / static_cast<double>(regions.size()));
    }
    analysis.regions = std::move(regions);
    return analysis;
}

}  // namespace anom::model
