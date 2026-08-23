#include "pipeline/spade.h"

#include "adapters/feature_utils.h"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace anom::model {

SpadePipeline::SpadePipeline(SpadePipelineConfig config) : config_(std::move(config)) {}

Result<void> SpadePipeline::addBackendOutputs(
    const TensorMap& outputs,
    const std::unordered_map<std::string, std::string>& outputBindings,
    const SPADEConfig& algorithmConfig) {
    for (const auto& semantic : algorithmConfig.featureLayers) {
        const auto binding = outputBindings.find(semantic);
        if (binding == outputBindings.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "SPADE feature semantic is not bound", semantic);
        }
        const auto tensor = outputs.find(binding->second);
        if (tensor == outputs.end()) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "Backend did not return SPADE feature tensor", binding->second);
        }
        auto view = feature::viewNchwFloat(tensor->second, "SPADE training feature");
        if (!view) return view.status();
        const std::vector<float> patches = feature::flattenPatches(tensor->second);
        for (const float value : patches) {
            if (!std::isfinite(value)) {
                return Status::error(ErrorCode::InvalidArgument,
                                     "SPADE training feature contains non-finite values",
                                     semantic);
            }
        }
        auto& accumulated = layerFeatures_[semantic];
        if (accumulated.size() > std::numeric_limits<std::size_t>::max() - patches.size()) {
            return Status::error(ErrorCode::OutOfMemory,
                                 "SPADE feature accumulation overflows size_t");
        }
        if (layerOrder_.empty() || layerOrder_.back() != semantic) layerOrder_.push_back(semantic);
        layerChannels_[semantic] = view.value().channels;
        accumulated.insert(accumulated.end(), patches.begin(), patches.end());
    }
    return {};
}

Result<void> SpadePipeline::saveIndexes(
    const std::vector<std::filesystem::path>& paths) const {
    if (paths.size() != layerOrder_.size()) {
        return Status::error(ErrorCode::InvalidArgument,
                             "SPADE index paths must align one-to-one with feature layers");
    }
    for (std::size_t i = 0; i < layerOrder_.size(); ++i) {
        const auto& semantic = layerOrder_[i];
        const auto features = layerFeatures_.find(semantic);
        if (features == layerFeatures_.end() || features->second.empty()) {
            return Status::error(ErrorCode::NotInitialized,
                                 "SPADE pipeline contains no training features for layer",
                                 semantic);
        }
        const int dimension = layerChannels_.at(semantic);
        const std::size_t count = features->second.size() / static_cast<std::size_t>(dimension);
        if (count > static_cast<std::size_t>(std::numeric_limits<faiss::idx_t>::max())) {
            return Status::error(ErrorCode::InvalidArgument,
                                 "SPADE layer gallery exceeds the FAISS index range",
                                 semantic);
        }
        try {
            faiss::IndexFlatL2 index(dimension);
            index.add(static_cast<faiss::idx_t>(count), features->second.data());
            faiss::write_index(&index, paths[i].string().c_str());
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::IoError,
                                 "Unable to save SPADE layer index", error.what());
        }
    }
    return {};
}

void SpadePipeline::clear() noexcept {
    layerOrder_.clear();
    layerFeatures_.clear();
    layerChannels_.clear();
}

std::size_t SpadePipeline::featureCount() const noexcept {
    std::size_t count = 0;
    for (const auto& [semantic, features] : layerFeatures_) {
        const int dimension = layerChannels_.count(semantic) ? layerChannels_.at(semantic) : 0;
        if (dimension > 0) count += features.size() / static_cast<std::size_t>(dimension);
    }
    return count;
}

}  // namespace anom::model
