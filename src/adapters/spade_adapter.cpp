#include "adapters/spade_adapter.h"

#include "adapters/feature_utils.h"
#include "retrieval/faiss_index.h"

#include <faiss/Index.h>
#include <faiss/index_io.h>

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

namespace anom::model {

class SpadeAdapter::Impl {
public:
    Impl(SPADEConfig config, std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage& package) {
        indexes_.clear();
        indexes_.reserve(config_.featureLayers.size());
        for (const auto& indexFile : config_.indexFiles) {
            auto path = package.resolveArtifact(indexFile);
            if (!path) return path.status();
            std::unique_ptr<faiss::Index> candidate;
            try {
                candidate.reset(faiss::read_index(path.value().string().c_str()));
            } catch (const std::exception& error) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "Unable to load SPADE FAISS index", error.what());
            }
            if (!candidate) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "FAISS returned a null SPADE index", path.value().string());
            }
            if (candidate->metric_type != faiss::METRIC_L2) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "SPADE adapter currently requires a FAISS L2 index");
            }
            if (candidate->ntotal < config_.numNeighbors) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "SPADE index contains fewer vectors than num_neighbors",
                                     std::to_string(candidate->ntotal));
            }
            indexes_.push_back(std::make_unique<FaissIndex>(std::move(candidate), path.value()));
        }
        return {};
    }

    Result<SearchExecutionInfo> configureSearch(const SearchExecutionConfig& config) {
        std::vector<FaissIndex*> indexes;
        for (auto& index : indexes_) indexes.push_back(index.get());
        return configureFaissIndexes(indexes, config, config_.numNeighbors);
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        std::set<std::string> semantics;
        std::set<std::string> tensorNames;
        for (const auto& semantic : config_.featureLayers) {
            if (!semantics.insert(semantic).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "SPADE feature semantic is duplicated", semantic);
            }
            const auto binding = outputBindings_.find(semantic);
            if (binding == outputBindings_.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "SPADE feature semantic is not bound", semantic);
            }
            if (!tensorNames.insert(binding->second).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "SPADE feature layers must bind distinct tensors",
                                     binding->second);
            }
            const auto* spec = signature.findOutput(binding->second);
            if (!spec || spec->dtype != DataType::Float32 || spec->shape.dims.size() != 4) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "SPADE feature must be a float32 NCHW output", binding->second);
            }
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& inputSize) const {
        if (indexes_.empty()) return Status::error(ErrorCode::NotInitialized, "SPADE indexes are not loaded");
        if (indexes_.size() != config_.featureLayers.size()) {
            return Status::error(ErrorCode::AdapterFailure, "SPADE index count does not match feature layers");
        }

        int batch = -1;
        for (const auto& semantic : config_.featureLayers) {
            const auto binding = outputBindings_.find(semantic);
            if (binding == outputBindings_.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "SPADE feature semantic is not bound", semantic);
            }
            const auto tensor = outputs.find(binding->second);
            if (tensor == outputs.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "Backend did not return SPADE feature tensor", binding->second);
            }
            auto view = feature::viewNchwFloat(tensor->second, "SPADE feature");
            if (!view) return view.status();
            if (batch < 0) batch = view.value().batch;
            if (view.value().batch != batch) {
                return Status::error(ErrorCode::TensorShapeMismatch,
                                     "SPADE feature tensors have inconsistent batch dimensions");
            }
        }
        if (batch <= 0) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "SPADE feature tensors have an empty batch dimension");
        }

        RawPredictionBatch result(static_cast<std::size_t>(batch));
        for (int n = 0; n < batch; ++n) {
            std::vector<cv::Mat> layerMaps;
            layerMaps.reserve(config_.featureLayers.size());
            for (std::size_t i = 0; i < config_.featureLayers.size(); ++i) {
                const auto binding = outputBindings_.find(config_.featureLayers[i]);
                const auto tensor = outputs.find(binding->second);
                auto view = feature::viewNchwFloat(tensor->second, "SPADE feature");
                const int locations = view.value().height * view.value().width;
                const int channels = view.value().channels;
                if (channels != indexes_[i]->dimension())
                    return Status::error(ErrorCode::TensorShapeMismatch,
                                         "SPADE feature channels do not match Faiss index dimension");
                const std::vector<float> patches = feature::flattenPatches(tensor->second);
                const float* queries = patches.data() +
                    static_cast<std::size_t>(n) * locations * channels;

                std::vector<float> distances(static_cast<std::size_t>(locations) * config_.numNeighbors);
                std::vector<faiss::idx_t> labels(static_cast<std::size_t>(locations) * config_.numNeighbors);
                try {
                    indexes_[i]->search(static_cast<faiss::idx_t>(locations), queries,
                                        config_.numNeighbors, distances.data(), labels.data());
                } catch (const std::exception& error) {
                    return Status::error(ErrorCode::AdapterFailure, "FAISS SPADE search failed", error.what());
                }
                std::vector<float> patchScores(static_cast<std::size_t>(locations));
                for (int location = 0; location < locations; ++location) {
                    double sum = 0.0;
                    for (int neighbor = 0; neighbor < config_.numNeighbors; ++neighbor) {
                        const float squared = distances[static_cast<std::size_t>(location) * config_.numNeighbors + neighbor];
                        if (!std::isfinite(squared) || squared < 0.0F) {
                            return Status::error(ErrorCode::AdapterFailure,
                                                 "FAISS produced an invalid SPADE neighbor distance");
                        }
                        sum += std::sqrt(squared);
                    }
                    patchScores[static_cast<std::size_t>(location)] =
                        static_cast<float>(sum / config_.numNeighbors);
                }
                layerMaps.push_back(feature::makeAnomalyMap(
                    patchScores.data(), view.value().height, view.value().width, inputSize, 0.0F));
            }

            cv::Mat fused(inputSize, CV_32FC1, cv::Scalar(0.0F));
            for (const auto& map : layerMaps) fused += map;
            fused /= static_cast<float>(layerMaps.size());
            if (config_.gaussianSigma > 0.0F) {
                cv::GaussianBlur(fused, fused, cv::Size(0, 0),
                                 config_.gaussianSigma, config_.gaussianSigma);
            }
            double minimum = 0.0, maximum = 0.0;
            cv::minMaxLoc(fused, &minimum, &maximum);
            result[static_cast<std::size_t>(n)].score = static_cast<float>(maximum);
            result[static_cast<std::size_t>(n)].anomalyMap = fused;
        }
        return result;
    }

private:
    SPADEConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    std::vector<std::unique_ptr<FaissIndex>> indexes_;
};

SpadeAdapter::SpadeAdapter(SPADEConfig config,
                           std::unordered_map<std::string, std::string> outputBindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(outputBindings))) {}
SpadeAdapter::~SpadeAdapter() = default;
Result<void> SpadeAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<SearchExecutionInfo> SpadeAdapter::configureSearch(const SearchExecutionConfig& config) {
    return impl_->configureSearch(config);
}
Result<void> SpadeAdapter::validateSignature(const TensorSignature& signature) const {
    return impl_->validateSignature(signature);
}
Result<RawPredictionBatch> SpadeAdapter::predict(const TensorMap& outputs,
                                                 const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
