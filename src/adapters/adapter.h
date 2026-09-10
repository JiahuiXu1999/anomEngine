#pragma once

#include "model/manifest.h"
#include "model/result.h"
#include "model/types.h"
#include "retrieval/search_execution.h"

#include <memory>

namespace anom::model {

class IModelAdapter {
public:
    virtual ~IModelAdapter() = default;
    virtual Result<void> loadAssets(const ModelPackage& package) = 0;
    // Called only during load, before prediction. Non-Faiss algorithms have no search stage.
    virtual Result<SearchExecutionInfo> configureSearch(const SearchExecutionConfig&) {
        return SearchExecutionInfo{};
    }
    virtual Result<void> validateSignature(const TensorSignature& signature) const = 0;
    virtual Result<RawPredictionBatch> predict(
        const TensorMap& outputs,
        const cv::Size& modelInputSize) const = 0;
};

Result<std::unique_ptr<IModelAdapter>> createAdapter(
    const ModelPackage& package,
    const std::filesystem::path& pluginDirectory = {});

}  // namespace anom::model
