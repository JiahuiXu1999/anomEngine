#include "processing/postprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

namespace anom::model {

float AnomalyPostprocessor::normalize(float value, float threshold, float minimum, float maximum) const {
    if (!config_.normalize) return value;
    const float normalized = (value - threshold) / (maximum - minimum) + 0.5F;
    return std::clamp(normalized, 0.0F, 1.0F);
}

cv::Mat AnomalyPostprocessor::restoreGeometry(const cv::Mat& map, const ImageGeometry& geometry) const {
    if (map.empty()) return {};
    cv::Mat floatMap;
    if (map.type() == CV_32FC1) floatMap = map;
    else map.convertTo(floatMap, CV_32FC1);

    cv::Mat resizedCanvas(geometry.resizedSize, CV_32FC1, cv::Scalar(0.0F));
    cv::Mat cropMap;
    if (floatMap.size() == geometry.cropRect.size()) cropMap = floatMap;
    else cv::resize(floatMap, cropMap, geometry.cropRect.size(), 0.0, 0.0, cv::INTER_LINEAR);
    cropMap.copyTo(resizedCanvas(geometry.cropRect));

    cv::Mat restored;
    cv::resize(resizedCanvas, restored, geometry.originalSize, 0.0, 0.0, cv::INTER_LINEAR);
    return restored;
}

Result<void> AnomalyPostprocessor::smooth(cv::Mat& map) const {
    if (!config_.smooth || map.empty()) return {};
    const cv::Size kernel(config_.smoothKernel, config_.smoothKernel);
    const double sigma = config_.smoothSigma > 0.0F ? static_cast<double>(config_.smoothSigma) : 0.0;
    cv::GaussianBlur(map, map, kernel, sigma, sigma, cv::BORDER_REPLICATE);
    return {};
}

Result<void> AnomalyPostprocessor::morph(cv::Mat& mask) const {
    if (!config_.morphology || mask.empty()) return {};
    const int operation = config_.morphologyMode == "close" ? cv::MORPH_CLOSE : cv::MORPH_OPEN;
    const cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_RECT, cv::Size(config_.morphologyKernel, config_.morphologyKernel));
    cv::morphologyEx(mask, mask, operation, kernel, cv::Point(-1, -1),
                     config_.morphologyIterations, cv::BORDER_CONSTANT, cv::morphologyDefaultBorderValue());
    return {};
}

Result<PredictionBatch> AnomalyPostprocessor::process(
    const RawPredictionBatch& raw, const std::vector<ImageGeometry>& geometry) const {
    if (raw.size() != geometry.size()) {
        return Status::error(ErrorCode::InternalError,
                             "Raw prediction count does not match input geometry count");
    }

    PredictionBatch output;
    output.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (!std::isfinite(raw[i].score)) {
            return Status::error(ErrorCode::AdapterFailure, "Adapter produced a non-finite image score",
                                 "batch index " + std::to_string(i));
        }
        Prediction prediction;
        prediction.modelId = modelId_;
        prediction.modelVersion = modelVersion_;
        prediction.rawScore = raw[i].score;
        prediction.score = normalize(raw[i].score, config_.imageThreshold,
                                     config_.imageMin, config_.imageMax);
        const float imageDecisionThreshold = config_.normalize
            ? 1.0F - config_.imageSensitivity
            : config_.imageThreshold;
        prediction.isAnomalous = config_.threshold && prediction.score >= imageDecisionThreshold;

        if (!raw[i].anomalyMap.empty()) {
            prediction.rawAnomalyMap = restoreGeometry(raw[i].anomalyMap, geometry[i]);
            prediction.anomalyMap.create(prediction.rawAnomalyMap.size(), CV_32FC1);
            for (int y = 0; y < prediction.rawAnomalyMap.rows; ++y) {
                const float* source = prediction.rawAnomalyMap.ptr<float>(y);
                float* target = prediction.anomalyMap.ptr<float>(y);
                for (int x = 0; x < prediction.rawAnomalyMap.cols; ++x) {
                    target[x] = normalize(source[x], config_.pixelThreshold,
                                          config_.pixelMin, config_.pixelMax);
                }
            }

            auto smoothed = smooth(prediction.anomalyMap);
            if (!smoothed) return smoothed.status();

            if (config_.threshold) {
                const float pixelDecisionThreshold = config_.normalize
                    ? 1.0F - config_.pixelSensitivity
                    : config_.pixelThreshold;
                cv::compare(prediction.anomalyMap, pixelDecisionThreshold, prediction.mask, cv::CMP_GE);
                auto morphed = morph(prediction.mask);
                if (!morphed) return morphed.status();
            }

            if (config_.analyze) {
                auto analyzed = analyzer_.analyze(prediction.anomalyMap, prediction.mask);
                if (!analyzed) return analyzed.status();
                prediction.analysis = std::move(analyzed.value());
            }
        }
        output.push_back(std::move(prediction));
    }
    return output;
}

}  // namespace anom::model
