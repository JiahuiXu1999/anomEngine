#pragma once

#include "adapters/adapter.h"

#include <memory>

namespace anom::model {

// Inference-side YOLO adapter. It decodes the raw anchor-free detection head
// (e.g. YOLOv8 [N, 4+C, M]): boxes are read as absolute-pixel (cx, cy, w, h),
// per-class sigmoid scores are collapsed into a confidence, then confidence
// filtering and non-maximum suppression produce the final detections. A
// detected box means "anomalous" — the highest confidence becomes the image
// score and the box regions are painted into the anomaly map.
class YoloAdapter final : public IModelAdapter {
public:
    YoloAdapter(YoloConfig config,
                std::unordered_map<std::string, std::string> outputBindings);
    ~YoloAdapter() override;
    YoloAdapter(const YoloAdapter&) = delete;
    YoloAdapter& operator=(const YoloAdapter&) = delete;

    Result<void> loadAssets(const ModelPackage& package) override;
    Result<void> validateSignature(const TensorSignature& signature) const override;
    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& modelInputSize) const override;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace anom::model
