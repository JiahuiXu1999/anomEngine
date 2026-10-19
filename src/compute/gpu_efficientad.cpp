#include "compute/gpu_compute.h"

#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace anom::model {
namespace {
void requireEfficientAd(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}

int efficientAdDimension(int64_t value) {
    requireEfficientAd(value > 0 && value <= INT_MAX, "Invalid EfficientAD tensor dimension");
    return static_cast<int>(value);
}

void validateEfficientAdTensor(const Tensor& tensor) {
    const auto count = tensor.shape.elementCount();
    requireEfficientAd(tensor.dtype == DataType::Float32 && count.ok() && count.value() > 0 &&
                          count.value() <= SIZE_MAX / sizeof(float) &&
                          tensor.byteSize() == count.value() * sizeof(float),
                      "EfficientAD requires a complete float32 tensor");
    const auto view = GpuCompute::buffer(tensor);
    requireEfficientAd(view.data != nullptr, "EfficientAD tensor data is null");
}

anom_compute_layer efficientAdLayer(const Tensor& tensor) {
    requireEfficientAd(tensor.shape.dims.size() == 4, "EfficientAD features must be NCHW");
    validateEfficientAdTensor(tensor);
    return {GpuCompute::buffer(tensor), efficientAdDimension(tensor.shape.dims[0]),
            efficientAdDimension(tensor.shape.dims[1]), efficientAdDimension(tensor.shape.dims[2]),
            efficientAdDimension(tensor.shape.dims[3])};
}

std::vector<int64_t> efficientAdMapShape(const Tensor& tensor) {
    const auto& dimensions = tensor.shape.dims;
    requireEfficientAd(dimensions.size() == 3 || (dimensions.size() == 4 && dimensions[1] == 1),
                      "EfficientAD map must be NHW or N1HW");
    validateEfficientAdTensor(tensor);
    return {efficientAdDimension(dimensions[0]),
            efficientAdDimension(dimensions[dimensions.size() - 2]),
            efficientAdDimension(dimensions.back())};
}
}  // namespace

Tensor GpuCompute::channelL2(const Tensor& input, const Tensor* reference) {
    const auto layer = efficientAdLayer(input);
    anom_compute_layer other{};
    if (reference) {
        other = efficientAdLayer(*reference);
        requireEfficientAd(input.shape.dims == reference->shape.dims,
                          "EfficientAD channel difference requires equal shapes");
    }
    auto result = allocate({layer.n, 1, layer.h, layer.w});
    anom_compute_channel_l2 args{layer, other, reference ? 1 : 0, buffer(result)};
    run(ANOM_COMPUTE_CHANNEL_L2, args);
    return result;
}

Tensor GpuCompute::normalizedBlend(const Tensor& hard, const Tensor& soft, float hardWeight,
                                   float softWeight) {
    const auto dimensions = efficientAdMapShape(hard);
    requireEfficientAd(dimensions == efficientAdMapShape(soft), "EfficientAD map shapes differ");
    const float total = hardWeight + softWeight;
    requireEfficientAd(std::isfinite(hardWeight) && hardWeight >= 0 && std::isfinite(softWeight) &&
                          softWeight >= 0 && std::isfinite(total) && total > 0,
                      "EfficientAD weights must have a positive finite sum");
    auto result = allocate(dimensions);
    anom_compute_normalized_blend args{buffer(hard), buffer(soft), buffer(result),
                                       static_cast<int>(dimensions[0]), static_cast<int>(dimensions[1]),
                                       static_cast<int>(dimensions[2]), hardWeight, softWeight};
    run(ANOM_COMPUTE_NORMALIZED_BLEND, args);
    return result;
}
}  // namespace anom::model
