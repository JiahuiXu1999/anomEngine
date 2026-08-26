#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"

#include <memory>

namespace anom::model {

class IModelAdapter {
public:
    virtual ~IModelAdapter() = default;
    virtual Result<void> loadAssets(const ModelPackage& package) = 0;
    virtual Result<void> validateSignature(const TensorSignature& signature) const = 0;
    virtual Result<RawPredictionBatch> predict(
        const TensorMap& outputs,
        const cv::Size& modelInputSize) const = 0;
};

Result<std::unique_ptr<IModelAdapter>> createAdapter(
    const ModelPackage& package,
    const std::filesystem::path& pluginDirectory = {});

}  // namespace anom::model
