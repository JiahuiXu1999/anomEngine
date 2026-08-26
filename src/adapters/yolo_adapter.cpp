#include "adapters/yolo_adapter.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>
#include <vector>

namespace anom::model {
namespace {

struct Detection {
    float x1{0.0F};
    float y1{0.0F};
    float x2{0.0F};
    float y2{0.0F};
    float score{0.0F};
    int classId{0};
};

float intersectionOverUnion(const Detection& a, const Detection& b) {
    const float x1 = std::max(a.x1, b.x1);
    const float y1 = std::max(a.y1, b.y1);
    const float x2 = std::min(a.x2, b.x2);
    const float y2 = std::min(a.y2, b.y2);
    const float intersection = std::max(0.0F, x2 - x1) * std::max(0.0F, y2 - y1);
    const float areaA = std::max(0.0F, a.x2 - a.x1) * std::max(0.0F, a.y2 - a.y1);
    const float areaB = std::max(0.0F, b.x2 - b.x1) * std::max(0.0F, b.y2 - b.y1);
    const float unionArea = areaA + areaB - intersection;
    return unionArea > 0.0F ? intersection / unionArea : 0.0F;
}

}  // namespace

class YoloAdapter::Impl {
public:
    Impl(YoloConfig config, std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage&) {
        // A supervised detector carries no auxiliary algorithm artifact — the
        // model weights themselves are the only asset to load.
        return {};
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        const auto binding = outputBindings_.find(config_.detectionSemantic);
        if (binding == outputBindings_.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "YOLO detection semantic is not bound", config_.detectionSemantic);
        }
        const auto* spec = signature.findOutput(binding->second);
        if (!spec || spec->dtype != DataType::Float32 || spec->shape.dims.size() != 3) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "YOLO detection output must be a float32 rank-3 tensor",
                                 binding->second);
        }
        const std::int64_t channels = spec->shape.dims[1];
        const std::int64_t expected = 4 + static_cast<std::int64_t>(config_.numClasses);
        if (channels > 0 && channels != expected) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "YOLO detection channels do not match num_classes",
                                 std::to_string(channels) + " != " + std::to_string(expected));
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& inputSize) const {
        const auto binding = outputBindings_.find(config_.detectionSemantic);
        if (binding == outputBindings_.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "YOLO detection semantic is not bound", config_.detectionSemantic);
        }
        const auto tensor = outputs.find(binding->second);
        if (tensor == outputs.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "Backend did not return YOLO detection output", binding->second);
        }
        const Tensor& detections = tensor->second;
        if (detections.dtype != DataType::Float32 || detections.shape.dims.size() != 3) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "YOLO detection output must be a float32 rank-3 tensor",
                                 detections.shape.toString());
        }
        const std::int64_t batchDim = detections.shape.dims[0];
        const std::int64_t channelDim = detections.shape.dims[1];
        const std::int64_t anchorDim = detections.shape.dims[2];
        const std::int64_t expectedChannels = 4 + static_cast<std::int64_t>(config_.numClasses);
        if (batchDim <= 0 || anchorDim <= 0 || channelDim != expectedChannels) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "YOLO detection shape does not match num_classes",
                                 detections.shape.toString());
        }
        const auto count = detections.shape.elementCount();
        if (!count) return count.status();
        if (count.value() * sizeof(float) != detections.byteSize()) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "YOLO detection byte size does not match shape",
                                 detections.shape.toString());
        }
        const std::int64_t maxInt = std::numeric_limits<int>::max();
        if (batchDim > maxInt || anchorDim > maxInt || channelDim > maxInt) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "YOLO detection dimensions exceed the int range",
                                 detections.shape.toString());
        }

        const int batch = static_cast<int>(batchDim);
        const int channels = static_cast<int>(channelDim);
        const int numAnchors = static_cast<int>(anchorDim);
        const float* data = detections.data<float>();

        RawPredictionBatch result(static_cast<std::size_t>(batch));
        for (int n = 0; n < batch; ++n) {
            const float* batchData = data + static_cast<std::size_t>(n) * channels * numAnchors;
            std::vector<Detection> candidates;
            for (int m = 0; m < numAnchors; ++m) {
                const float cx = batchData[0 * numAnchors + m];
                const float cy = batchData[1 * numAnchors + m];
                const float w = batchData[2 * numAnchors + m];
                const float h = batchData[3 * numAnchors + m];
                float best = -std::numeric_limits<float>::infinity();
                int bestClass = 0;
                for (int c = 0; c < config_.numClasses; ++c) {
                    const float score = batchData[(4 + c) * numAnchors + m];
                    if (score > best) {
                        best = score;
                        bestClass = c;
                    }
                }
                if (best < config_.confThreshold) continue;
                candidates.push_back(
                    Detection{cx - w * 0.5F, cy - h * 0.5F, cx + w * 0.5F, cy + h * 0.5F,
                              best, bestClass});
            }

            std::sort(candidates.begin(), candidates.end(),
                      [](const Detection& a, const Detection& b) { return a.score > b.score; });
            std::vector<Detection> kept;
            kept.reserve(candidates.size());
            for (const auto& candidate : candidates) {
                bool suppressed = false;
                for (const auto& existing : kept) {
                    if (existing.classId == candidate.classId &&
                        intersectionOverUnion(existing, candidate) > config_.nmsThreshold) {
                        suppressed = true;
                        break;
                    }
                }
                if (!suppressed) kept.push_back(candidate);
            }

            float imageScore = 0.0F;
            cv::Mat map = cv::Mat::zeros(inputSize, CV_32FC1);
            for (const auto& detection : kept) {
                imageScore = std::max(imageScore, detection.score);
                const int x1 = std::max(0, static_cast<int>(std::floor(detection.x1)));
                const int y1 = std::max(0, static_cast<int>(std::floor(detection.y1)));
                const int x2 = std::min(inputSize.width, static_cast<int>(std::ceil(detection.x2)));
                const int y2 = std::min(inputSize.height, static_cast<int>(std::ceil(detection.y2)));
                if (x2 <= x1 || y2 <= y1) continue;
                cv::rectangle(map, cv::Rect(x1, y1, x2 - x1, y2 - y1),
                              cv::Scalar(detection.score), cv::FILLED);
            }
            result[static_cast<std::size_t>(n)].score = imageScore;
            result[static_cast<std::size_t>(n)].anomalyMap = std::move(map);
        }
        return result;
    }

private:
    YoloConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
};

YoloAdapter::YoloAdapter(YoloConfig config,
                         std::unordered_map<std::string, std::string> outputBindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(outputBindings))) {}
YoloAdapter::~YoloAdapter() = default;
Result<void> YoloAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<void> YoloAdapter::validateSignature(const TensorSignature& signature) const {
    return impl_->validateSignature(signature);
}
Result<RawPredictionBatch> YoloAdapter::predict(const TensorMap& outputs,
                                                const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
