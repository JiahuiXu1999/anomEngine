#include "adapters/patchcore_adapter.h"

#include "adapters/feature_utils.h"

#include <faiss/Index.h>
#include <faiss/index_io.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <vector>

namespace anom::model {

class PatchCoreAdapter::Impl {
public:
    Impl(PatchCoreConfig config, std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage& package) {
        auto path = package.resolveArtifact(config_.indexFile);
        if (!path) return path.status();
        std::unique_ptr<faiss::Index> candidate;
        try {
            candidate.reset(faiss::read_index(path.value().string().c_str()));
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "Unable to load PatchCore FAISS index", error.what());
        }
        if (!candidate) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "FAISS returned a null index", path.value().string());
        }
        if (candidate->d != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "FAISS index dimension does not match manifest",
                                 std::to_string(candidate->d) + " != " +
                                     std::to_string(config_.embeddingDimension));
        }
        if (candidate->ntotal < config_.numNeighbors) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "FAISS index contains fewer vectors than num_neighbors",
                                 std::to_string(candidate->ntotal));
        }
        if (candidate->metric_type != faiss::METRIC_L2) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PatchCore adapter currently requires a FAISS L2 index");
        }
        index_ = std::move(candidate);
        return {};
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        std::set<std::string> semantics;
        std::set<std::string> tensorNames;
        std::int64_t staticBatch = -1;
        std::int64_t staticChannels = 0;
        bool allChannelsStatic = true;
        for (const auto& semantic : config_.featureLayers) {
            if (!semantics.insert(semantic).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PatchCore feature semantic is duplicated", semantic);
            }
            const auto binding = outputBindings_.find(semantic);
            if (binding == outputBindings_.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PatchCore feature semantic is not bound", semantic);
            }
            if (!tensorNames.insert(binding->second).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PatchCore feature layers must bind distinct tensors",
                                     binding->second);
            }
            const auto* spec = signature.findOutput(binding->second);
            if (!spec) return Status::error(ErrorCode::TensorSignatureMismatch,
                                            "PatchCore feature tensor is absent from backend", binding->second);
            if (spec->dtype != DataType::Float32 || spec->shape.dims.size() != 4) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PatchCore feature must be a float32 NCHW tensor", binding->second);
            }
            const auto& dims = spec->shape.dims;
            for (const auto dim : dims) {
                if (dim == 0 || dim < -1) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "PatchCore feature has an invalid dimension",
                                         binding->second + " " + spec->shape.toString());
                }
            }
            if (dims[0] > 0) {
                if (staticBatch > 0 && staticBatch != dims[0]) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "PatchCore feature tensors have incompatible batch dimensions",
                                         binding->second + " " + spec->shape.toString());
                }
                staticBatch = dims[0];
            }
            if (dims[1] > 0) {
                staticChannels += dims[1];
            } else {
                allChannelsStatic = false;
            }

            if (dims[2] > 0 && dims[3] > 0) {
                const auto pooledHeight =
                    (dims[2] + 2LL * config_.poolingPadding - config_.poolingKernel) /
                        config_.poolingStride +
                    1;
                const auto pooledWidth =
                    (dims[3] + 2LL * config_.poolingPadding - config_.poolingKernel) /
                        config_.poolingStride +
                    1;
                if (pooledHeight <= 0 || pooledWidth <= 0) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "PatchCore pooling produces an empty feature map",
                                         binding->second + " " + spec->shape.toString());
                }
            }
        }
        if (allChannelsStatic && staticChannels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "PatchCore feature channels do not match the embedding dimension",
                                 std::to_string(staticChannels) + " != " +
                                     std::to_string(config_.embeddingDimension));
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs, const cv::Size& inputSize) const {
        if (!index_) return Status::error(ErrorCode::NotInitialized, "PatchCore index is not loaded");
        auto embedding = feature::aggregateFeaturePyramid(
            outputs, outputBindings_, config_.featureLayers, true,
            config_.poolingKernel, config_.poolingStride, config_.poolingPadding);
        if (!embedding) return embedding.status();
        auto view = feature::viewNchwFloat(embedding.value(), "PatchCore embedding");
        if (!view) return view.status();
        if (view.value().channels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Aggregated PatchCore embedding dimension does not match manifest",
                                 std::to_string(view.value().channels));
        }

        const std::vector<float> queries = feature::flattenPatches(embedding.value());
        const std::size_t locations = static_cast<std::size_t>(view.value().height) * view.value().width;
        const std::size_t queryCount = static_cast<std::size_t>(view.value().batch) * locations;
        if (queryCount > static_cast<std::size_t>(std::numeric_limits<faiss::idx_t>::max())) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "PatchCore query count exceeds the FAISS index range",
                                 std::to_string(queryCount));
        }
        std::vector<float> distances(queryCount * config_.numNeighbors);
        std::vector<faiss::idx_t> indices(queryCount * config_.numNeighbors);
        try {
            index_->search(static_cast<faiss::idx_t>(queryCount), queries.data(), config_.numNeighbors,
                           distances.data(), indices.data());
        } catch (const std::exception& error) {
            return Status::error(ErrorCode::AdapterFailure, "FAISS PatchCore search failed", error.what());
        }

        for (std::size_t i = 0; i < distances.size(); ++i) {
            if (indices[i] < 0 || !std::isfinite(distances[i]) || distances[i] < 0.0F) {
                return Status::error(ErrorCode::AdapterFailure,
                                     "FAISS produced an invalid neighbor result", std::to_string(i));
            }
            if (config_.sqrtDistances) distances[i] = std::sqrt(distances[i]);
        }

        RawPredictionBatch result(static_cast<std::size_t>(view.value().batch));
        std::vector<float> patchScores(locations);
        for (int n = 0; n < view.value().batch; ++n) {
            float maximum = -std::numeric_limits<float>::infinity();
            std::size_t maximumLocation = 0;
            for (std::size_t location = 0; location < locations; ++location) {
                const std::size_t query = static_cast<std::size_t>(n) * locations + location;
                patchScores[location] = distances[query * config_.numNeighbors];
                if (patchScores[location] > maximum) {
                    maximum = patchScores[location];
                    maximumLocation = location;
                }
            }
            float imageScore = maximum;
            if (config_.weightedImageScore && config_.numNeighbors > 1) {
                const std::size_t query = static_cast<std::size_t>(n) * locations + maximumLocation;
                const auto supportIndex = indices[query * config_.numNeighbors];
                std::vector<float> support(static_cast<std::size_t>(config_.embeddingDimension));
                std::vector<float> supportDistances(static_cast<std::size_t>(config_.numNeighbors));
                std::vector<faiss::idx_t> supportIndices(static_cast<std::size_t>(config_.numNeighbors));
                try {
                    index_->reconstruct(supportIndex, support.data());
                    index_->search(1, support.data(), config_.numNeighbors,
                                   supportDistances.data(), supportIndices.data());
                } catch (const std::exception& error) {
                    return Status::error(ErrorCode::AdapterFailure,
                                         "FAISS index does not support PatchCore weighted scoring",
                                         error.what());
                }
                for (std::size_t neighbor = 0; neighbor < supportDistances.size(); ++neighbor) {
                    auto& distance = supportDistances[neighbor];
                    if (supportIndices[neighbor] < 0 || !std::isfinite(distance) || distance < 0.0F) {
                        return Status::error(ErrorCode::AdapterFailure,
                                             "FAISS produced an invalid PatchCore support neighbor",
                                             std::to_string(neighbor));
                    }
                    if (config_.sqrtDistances) distance = std::sqrt(distance);
                }
                const float maxDistance = *std::max_element(supportDistances.begin(), supportDistances.end());
                float denominator = 0.0F;
                for (const auto distance : supportDistances) denominator += std::exp(distance - maxDistance);
                const float nearestProbability = std::exp(supportDistances.front() - maxDistance) / denominator;
                imageScore = (1.0F - nearestProbability) * maximum;
            }
            result[static_cast<std::size_t>(n)].score = imageScore;
            result[static_cast<std::size_t>(n)].anomalyMap = feature::makeAnomalyMap(
                patchScores.data(), view.value().height, view.value().width,
                inputSize, config_.gaussianSigma);
        }
        return result;
    }

private:
    PatchCoreConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    std::unique_ptr<faiss::Index> index_;
};

PatchCoreAdapter::PatchCoreAdapter(PatchCoreConfig config,
                                   std::unordered_map<std::string, std::string> bindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(bindings))) {}
PatchCoreAdapter::~PatchCoreAdapter() = default;
Result<void> PatchCoreAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<void> PatchCoreAdapter::validateSignature(const TensorSignature& signature) const { return impl_->validateSignature(signature); }
Result<RawPredictionBatch> PatchCoreAdapter::predict(const TensorMap& outputs,
                                                     const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
