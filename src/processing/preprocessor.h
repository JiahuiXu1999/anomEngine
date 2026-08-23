#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <opencv2/core.hpp>

#include <vector>

namespace anom::model {

class ImagePreprocessor {
public:
    explicit ImagePreprocessor(InputConfig config) : config_(std::move(config)) {}

    Result<PreprocessedBatch> process(const std::vector<cv::Mat>& images) const;
    [[nodiscard]] cv::Size modelInputSize() const noexcept;

private:
    InputConfig config_;
};

}  // namespace anom::model
