#pragma once

#include "model/result.h"
#include "model/types.h"

#include <string>
#include <vector>

namespace anom::model::feature {

struct NchwView {
    const float* data{nullptr};
    int batch{0};
    int channels{0};
    int height{0};
    int width{0};
};

Result<NchwView> viewNchwFloat(const Tensor& tensor, const std::string& context);
Result<Tensor> averagePool2d(const Tensor& input, int kernel, int stride, int padding);
Result<Tensor> resizeNchw(const Tensor& input, int height, int width);
Result<Tensor> concatenateChannels(const std::vector<Tensor>& inputs);
Result<Tensor> selectChannels(const Tensor& input, const std::vector<std::int32_t>& indices);

// Collapse an NCHW float32 tensor to a single-channel Nx1xHxW map of the
// per-location L2 norm across channels. Used for EfficientAD soft scoring.
Result<Tensor> channelWiseL2(const Tensor& input);

// Per-location L2 distance across channels between two equally shaped NCHW
// float32 tensors, returned as a single-channel Nx1xHxW map. Used for the
// EfficientAD teacher/student hard score.
Result<Tensor> channelWiseL2Difference(const Tensor& left, const Tensor& right);

Result<Tensor> aggregateFeaturePyramid(
    const TensorMap& tensors,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const std::vector<std::string>& semantics,
    bool applyAveragePooling,
    int poolingKernel,
    int poolingStride,
    int poolingPadding);

std::vector<float> flattenPatches(const Tensor& nchw);
cv::Mat makeAnomalyMap(const float* scores, int height, int width,
                       const cv::Size& outputSize, float gaussianSigma);

}  // namespace anom::model::feature
