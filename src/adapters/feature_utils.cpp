#include "adapters/feature_utils.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>

namespace anom::model::feature {
namespace {

Tensor allocateFloatNchw(int batch, int channels, int height, int width) {
    Tensor tensor;
    tensor.dtype = DataType::Float32;
    tensor.shape.dims = {batch, channels, height, width};
    tensor.bytes.resize(static_cast<std::size_t>(batch) * channels * height * width * sizeof(float));
    return tensor;
}

}  // namespace

Result<NchwView> viewNchwFloat(const Tensor& tensor, const std::string& context) {
    if (tensor.dtype != DataType::Float32) {
        return Status::error(ErrorCode::TensorTypeMismatch,
                             "Feature tensor must use float32", context);
    }
    if (tensor.shape.dims.size() != 4) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "Feature tensor must have NCHW rank 4", context + " " + tensor.shape.toString());
    }
    for (const auto dim : tensor.shape.dims) {
        if (dim <= 0 || dim > std::numeric_limits<int>::max()) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Feature tensor dimensions must be positive int32 values", context);
        }
    }
    const auto count = tensor.shape.elementCount();
    if (!count) return count.status();
    if (count.value() * sizeof(float) != tensor.byteSize()) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "Feature tensor byte size does not match shape", context);
    }
    return NchwView{tensor.data<float>(),
                    static_cast<int>(tensor.shape.dims[0]),
                    static_cast<int>(tensor.shape.dims[1]),
                    static_cast<int>(tensor.shape.dims[2]),
                    static_cast<int>(tensor.shape.dims[3])};
}

Result<Tensor> averagePool2d(const Tensor& input, int kernel, int stride, int padding) {
    auto source = viewNchwFloat(input, "averagePool2d");
    if (!source) return source.status();
    if (kernel <= 0 || stride <= 0 || padding < 0) {
        return Status::error(ErrorCode::InvalidArgument, "Invalid average pooling parameters");
    }
    const int outH = (source.value().height + 2 * padding - kernel) / stride + 1;
    const int outW = (source.value().width + 2 * padding - kernel) / stride + 1;
    if (outH <= 0 || outW <= 0) {
        return Status::error(ErrorCode::TensorShapeMismatch, "Average pooling produces an empty tensor");
    }
    Tensor output = allocateFloatNchw(source.value().batch, source.value().channels, outH, outW);
    float* target = output.data<float>();
    const float divisor = static_cast<float>(kernel * kernel);
    for (int n = 0; n < source.value().batch; ++n) {
        for (int c = 0; c < source.value().channels; ++c) {
            const float* plane = source.value().data +
                (static_cast<std::size_t>(n) * source.value().channels + c) * source.value().height * source.value().width;
            for (int oh = 0; oh < outH; ++oh) {
                for (int ow = 0; ow < outW; ++ow) {
                    float sum = 0.0F;
                    const int startH = oh * stride - padding;
                    const int startW = ow * stride - padding;
                    for (int kh = 0; kh < kernel; ++kh) {
                        const int ih = startH + kh;
                        if (ih < 0 || ih >= source.value().height) continue;
                        for (int kw = 0; kw < kernel; ++kw) {
                            const int iw = startW + kw;
                            if (iw >= 0 && iw < source.value().width) sum += plane[ih * source.value().width + iw];
                        }
                    }
                    const std::size_t index =
                        ((static_cast<std::size_t>(n) * source.value().channels + c) * outH + oh) * outW + ow;
                    target[index] = sum / divisor;
                }
            }
        }
    }
    return output;
}

Result<Tensor> resizeNchw(const Tensor& input, int height, int width) {
    auto source = viewNchwFloat(input, "resizeNchw");
    if (!source) return source.status();
    if (height <= 0 || width <= 0) return Status::error(ErrorCode::InvalidArgument, "Resize dimensions must be positive");
    if (height == source.value().height && width == source.value().width) return input;
    Tensor output = allocateFloatNchw(source.value().batch, source.value().channels, height, width);
    for (int n = 0; n < source.value().batch; ++n) {
        for (int c = 0; c < source.value().channels; ++c) {
            const std::size_t sourceOffset =
                (static_cast<std::size_t>(n) * source.value().channels + c) * source.value().height * source.value().width;
            const std::size_t targetOffset =
                (static_cast<std::size_t>(n) * source.value().channels + c) * height * width;
            cv::Mat sourcePlane(source.value().height, source.value().width, CV_32FC1,
                                const_cast<float*>(source.value().data + sourceOffset));
            cv::Mat targetPlane(height, width, CV_32FC1, output.data<float>() + targetOffset);
            cv::resize(sourcePlane, targetPlane, cv::Size(width, height), 0.0, 0.0, cv::INTER_LINEAR);
        }
    }
    return output;
}

Result<Tensor> concatenateChannels(const std::vector<Tensor>& inputs) {
    if (inputs.empty()) return Status::error(ErrorCode::InvalidArgument, "No feature tensors to concatenate");
    auto first = viewNchwFloat(inputs.front(), "concatenateChannels[0]");
    if (!first) return first.status();
    int channels = 0;
    std::vector<NchwView> views;
    views.reserve(inputs.size());
    for (std::size_t i = 0; i < inputs.size(); ++i) {
        auto view = viewNchwFloat(inputs[i], "concatenateChannels[" + std::to_string(i) + "]");
        if (!view) return view.status();
        if (view.value().batch != first.value().batch || view.value().height != first.value().height ||
            view.value().width != first.value().width) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Feature tensors must share batch and spatial dimensions before concatenation");
        }
        channels += view.value().channels;
        views.push_back(view.value());
    }
    Tensor output = allocateFloatNchw(first.value().batch, channels, first.value().height, first.value().width);
    const std::size_t planeElements = static_cast<std::size_t>(first.value().height) * first.value().width;
    for (int n = 0; n < first.value().batch; ++n) {
        int channelOffset = 0;
        for (const auto& view : views) {
            const float* source = view.data + static_cast<std::size_t>(n) * view.channels * planeElements;
            float* target = output.data<float>() +
                (static_cast<std::size_t>(n) * channels + channelOffset) * planeElements;
            std::memcpy(target, source, static_cast<std::size_t>(view.channels) * planeElements * sizeof(float));
            channelOffset += view.channels;
        }
    }
    return output;
}

Result<Tensor> selectChannels(const Tensor& input, const std::vector<std::int32_t>& indices) {
    auto source = viewNchwFloat(input, "selectChannels");
    if (!source) return source.status();
    if (indices.empty()) return Status::error(ErrorCode::InvalidArgument, "Channel selection must not be empty");
    Tensor output = allocateFloatNchw(source.value().batch, static_cast<int>(indices.size()),
                                      source.value().height, source.value().width);
    const std::size_t planeElements = static_cast<std::size_t>(source.value().height) * source.value().width;
    for (std::size_t c = 0; c < indices.size(); ++c) {
        const int selected = indices[c];
        if (selected < 0 || selected >= source.value().channels) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Selected channel index is outside aggregated embedding",
                                 std::to_string(selected));
        }
        for (int n = 0; n < source.value().batch; ++n) {
            const float* sourcePlane = source.value().data +
                (static_cast<std::size_t>(n) * source.value().channels + selected) * planeElements;
            float* targetPlane = output.data<float>() +
                (static_cast<std::size_t>(n) * indices.size() + c) * planeElements;
            std::memcpy(targetPlane, sourcePlane, planeElements * sizeof(float));
        }
    }
    return output;
}

Result<Tensor> channelWiseL2(const Tensor& input) {
    auto source = viewNchwFloat(input, "channelWiseL2");
    if (!source) return source.status();
    Tensor output = allocateFloatNchw(source.value().batch, 1, source.value().height,
                                      source.value().width);
    const std::size_t planeElements =
        static_cast<std::size_t>(source.value().height) * source.value().width;
    for (int n = 0; n < source.value().batch; ++n) {
        for (std::size_t location = 0; location < planeElements; ++location) {
            double sum = 0.0;
            for (int c = 0; c < source.value().channels; ++c) {
                const float value = source.value().data[
                    (static_cast<std::size_t>(n) * source.value().channels + c) * planeElements +
                    location];
                sum += static_cast<double>(value) * value;
            }
            output.data<float>()[static_cast<std::size_t>(n) * planeElements + location] =
                static_cast<float>(std::sqrt(sum));
        }
    }
    return output;
}

Result<Tensor> channelWiseL2Difference(const Tensor& left, const Tensor& right) {
    auto a = viewNchwFloat(left, "channelWiseL2Difference left");
    if (!a) return a.status();
    auto b = viewNchwFloat(right, "channelWiseL2Difference right");
    if (!b) return b.status();
    if (a.value().batch != b.value().batch || a.value().channels != b.value().channels ||
        a.value().height != b.value().height || a.value().width != b.value().width) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "channelWiseL2Difference requires equally shaped tensors",
                             left.shape.toString() + " vs " + right.shape.toString());
    }
    Tensor output = allocateFloatNchw(a.value().batch, 1, a.value().height, a.value().width);
    const std::size_t planeElements = static_cast<std::size_t>(a.value().height) * a.value().width;
    for (int n = 0; n < a.value().batch; ++n) {
        for (std::size_t location = 0; location < planeElements; ++location) {
            double sum = 0.0;
            for (int c = 0; c < a.value().channels; ++c) {
                const std::size_t index =
                    (static_cast<std::size_t>(n) * a.value().channels + c) * planeElements + location;
                const double difference =
                    static_cast<double>(a.value().data[index]) - b.value().data[index];
                sum += difference * difference;
            }
            output.data<float>()[static_cast<std::size_t>(n) * planeElements + location] =
                static_cast<float>(std::sqrt(sum));
        }
    }
    return output;
}

Result<Tensor> aggregateFeaturePyramid(
    const TensorMap& tensors,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const std::vector<std::string>& semantics,
    bool applyAveragePooling,
    int poolingKernel,
    int poolingStride,
    int poolingPadding) {
    std::vector<Tensor> features;
    features.reserve(semantics.size());
    int targetHeight = 0, targetWidth = 0;
    for (const auto& semantic : semantics) {
        const auto binding = outputBindings.find(semantic);
        if (binding == outputBindings.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "Feature semantic has no output binding", semantic);
        }
        const auto tensor = tensors.find(binding->second);
        if (tensor == tensors.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "Backend did not return required feature tensor", binding->second);
        }
        Tensor featureTensor = tensor->second;
        if (applyAveragePooling) {
            auto pooled = averagePool2d(featureTensor, poolingKernel, poolingStride, poolingPadding);
            if (!pooled) return pooled.status();
            featureTensor = std::move(pooled.value());
        }
        auto view = viewNchwFloat(featureTensor, semantic);
        if (!view) return view.status();
        if (features.empty()) { targetHeight = view.value().height; targetWidth = view.value().width; }
        if (view.value().height != targetHeight || view.value().width != targetWidth) {
            auto resized = resizeNchw(featureTensor, targetHeight, targetWidth);
            if (!resized) return resized.status();
            featureTensor = std::move(resized.value());
        }
        features.push_back(std::move(featureTensor));
    }
    return concatenateChannels(features);
}

std::vector<float> flattenPatches(const Tensor& nchw) {
    const int batch = static_cast<int>(nchw.shape.dims[0]);
    const int channels = static_cast<int>(nchw.shape.dims[1]);
    const int height = static_cast<int>(nchw.shape.dims[2]);
    const int width = static_cast<int>(nchw.shape.dims[3]);
    const std::size_t locations = static_cast<std::size_t>(height) * width;
    std::vector<float> output(static_cast<std::size_t>(batch) * locations * channels);
    const float* source = nchw.data<float>();
    for (int n = 0; n < batch; ++n) {
        for (std::size_t location = 0; location < locations; ++location) {
            float* patch = output.data() + (static_cast<std::size_t>(n) * locations + location) * channels;
            for (int c = 0; c < channels; ++c) {
                patch[c] = source[(static_cast<std::size_t>(n) * channels + c) * locations + location];
            }
        }
    }
    return output;
}

cv::Mat makeAnomalyMap(const float* scores, int height, int width,
                       const cv::Size& outputSize, float gaussianSigma) {
    cv::Mat lowResolution(height, width, CV_32FC1, const_cast<float*>(scores));
    cv::Mat map;
    cv::resize(lowResolution, map, outputSize, 0.0, 0.0, cv::INTER_LINEAR);
    if (gaussianSigma > 0.0F) cv::GaussianBlur(map, map, cv::Size(0, 0), gaussianSigma, gaussianSigma);
    return map;
}

}  // namespace anom::model::feature
