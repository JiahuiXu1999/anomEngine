#include "processing/preprocessor.h"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cstring>

namespace anom::model {

cv::Size ImagePreprocessor::modelInputSize() const noexcept {
    return config_.centerCrop.value_or(cv::Size(config_.resizeWidth, config_.resizeHeight));
}

Result<PreprocessedBatch> ImagePreprocessor::process(const std::vector<cv::Mat>& images) const {
    if (images.empty()) {
        return Status::error(ErrorCode::InvalidArgument, "Image batch must not be empty");
    }
    const cv::Size inputSize = modelInputSize();
    const int batch = static_cast<int>(images.size());
    const std::size_t imageElements = static_cast<std::size_t>(inputSize.width) * inputSize.height * 3;

    PreprocessedBatch output;
    output.tensor.dtype = DataType::Float32;
    output.tensor.shape.dims = config_.layout == TensorLayout::NCHW
        ? std::vector<std::int64_t>{batch, 3, inputSize.height, inputSize.width}
        : std::vector<std::int64_t>{batch, inputSize.height, inputSize.width, 3};
    output.tensor.bytes.resize(static_cast<std::size_t>(batch) * imageElements * sizeof(float));
    output.geometry.reserve(images.size());
    float* destination = output.tensor.data<float>();

    for (std::size_t n = 0; n < images.size(); ++n) {
        const cv::Mat& image = images[n];
        if (image.empty()) {
            return Status::error(ErrorCode::InvalidImage, "Input image is empty",
                                 "batch index " + std::to_string(n));
        }
        if (image.depth() != CV_8U) {
            return Status::error(ErrorCode::InvalidImage,
                                 "Only 8-bit input images are supported",
                                 "batch index " + std::to_string(n));
        }

        cv::Mat bgr;
        if (image.channels() == 3) bgr = image;
        else if (image.channels() == 1) cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
        else if (image.channels() == 4) cv::cvtColor(image, bgr, cv::COLOR_BGRA2BGR);
        else {
            return Status::error(ErrorCode::InvalidImage,
                                 "Input image must have 1, 3, or 4 channels",
                                 "batch index " + std::to_string(n));
        }

        cv::Mat resized;
        cv::resize(bgr, resized, cv::Size(config_.resizeWidth, config_.resizeHeight),
                   0.0, 0.0, config_.interpolation);

        cv::Rect cropRect(0, 0, resized.cols, resized.rows);
        cv::Mat cropped = resized;
        if (config_.centerCrop) {
            cropRect.x = (resized.cols - config_.centerCrop->width) / 2;
            cropRect.y = (resized.rows - config_.centerCrop->height) / 2;
            cropRect.width = config_.centerCrop->width;
            cropRect.height = config_.centerCrop->height;
            cropped = resized(cropRect);
        }

        cv::Mat ordered;
        if (config_.color == "RGB") cv::cvtColor(cropped, ordered, cv::COLOR_BGR2RGB);
        else ordered = cropped;
        cv::Mat floatImage;
        ordered.convertTo(floatImage, CV_32FC3, 1.0 / 255.0);

        const std::size_t batchOffset = n * imageElements;
        for (int h = 0; h < inputSize.height; ++h) {
            const auto* row = floatImage.ptr<cv::Vec3f>(h);
            for (int w = 0; w < inputSize.width; ++w) {
                for (int c = 0; c < 3; ++c) {
                    const float normalized = (row[w][c] - config_.mean[static_cast<std::size_t>(c)]) /
                                             config_.std[static_cast<std::size_t>(c)];
                    std::size_t offset;
                    if (config_.layout == TensorLayout::NCHW) {
                        offset = batchOffset + static_cast<std::size_t>(c) * inputSize.area() +
                                 static_cast<std::size_t>(h) * inputSize.width + w;
                    } else {
                        offset = batchOffset + (static_cast<std::size_t>(h) * inputSize.width + w) * 3 + c;
                    }
                    destination[offset] = normalized;
                }
            }
        }
        output.geometry.push_back(ImageGeometry{image.size(), resized.size(), cropRect});
    }
    return output;
}

}  // namespace anom::model
