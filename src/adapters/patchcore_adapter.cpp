#include "adapters/patchcore_adapter.h"
#include "compute/gpu_compute.h"

#include "adapters/feature_utils.h"
#include "retrieval/faiss_index.h"
#include "pipeline/patchcore_statistics.h"

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
        searchDimension_ = config_.embeddingDimension;
        if (config_.variant != PatchCoreVariant::Original) {
            auto statsPath = package.resolveArtifact(config_.statisticsFile);
            if (!statsPath) return statsPath.status();
            auto loaded = PatchCoreStatistics::load(statsPath.value(), config_.variant, config_.embeddingDimension);
            if (!loaded) return loaded.status();
            statistics_ = std::move(loaded.value());
            searchDimension_ = statistics_.basis.rows;
            if (config_.variant == PatchCoreVariant::SubspaceAD) return {};
        }
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
        if (candidate->d != searchDimension_) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "FAISS index dimension does not match manifest",
                                 std::to_string(candidate->d) + " != " +
                                     std::to_string(searchDimension_));
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
        index_ = std::make_unique<FaissIndex>(std::move(candidate), path.value());
        return {};
    }

    Result<SearchExecutionInfo> configureSearch(const SearchExecutionConfig& config) {
        if (config_.variant == PatchCoreVariant::SubspaceAD) return SearchExecutionInfo{};
        if (!index_) return Status::error(ErrorCode::NotInitialized, "PatchCore index is not loaded");
        return configureFaissIndexes({index_.get()}, config, config_.numNeighbors);
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        std::set<std::string> semantics;
        std::set<std::string> tensorNames;
        std::int64_t staticBatch = -1;
        std::int64_t staticChannels = 0;
        bool allChannelsStatic = true;
        std::vector<std::int64_t> meanShape;
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
            if (config_.meanAggregation) {
                if (meanShape.empty()) meanShape = dims;
                else for (std::size_t axis = 1; axis < dims.size(); ++axis) {
                    if (dims[axis] > 0 && meanShape[axis] > 0 && dims[axis] != meanShape[axis])
                        return Status::error(ErrorCode::TensorSignatureMismatch,
                                             "Mean aggregation requires matching feature shapes");
                }
            }
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
                if (config_.meanAggregation) staticChannels = dims[1];
                else staticChannels += dims[1];
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

    Result<void> configureCompute(int device, const std::filesystem::path& directory) {
        if (device < 0) { compute_.reset(); return {}; }
        auto loaded = GpuCompute::load(device, directory);
        if (!loaded) return loaded.status();
        compute_ = std::move(loaded.value());
        return {};
    }

    Result<RawPredictionBatch> predictGpu(const TensorMap& outputs, const cv::Size& inputSize) const {
        try {
            auto patches = compute_->aggregate(outputs, outputBindings_, config_);
            const int n = static_cast<int>(patches.shape.dims[0]);
            const int h = static_cast<int>(patches.shape.dims[1]), w = static_cast<int>(patches.shape.dims[2]);
            const int locations = h * w;
            const int64_t count = static_cast<int64_t>(n) * locations;
            Tensor patchScores;
            std::vector<float> imageScores;
            std::vector<int64_t> maxima;
            if (config_.variant == PatchCoreVariant::SubspaceAD) {
                patchScores = compute_->transform(patches, statistics_.mean, statistics_.basis, statistics_.scales, true);
                // Small image-level top-fraction selection stays on CPU.
                const auto host = compute_->download(patchScores);
                imageScores.resize(n);
                const auto top = std::max(1, static_cast<int>(std::ceil(locations * static_cast<double>(config_.topScoreFraction))));
                for (int i = 0; i < n; ++i) {
                    const auto* begin = host.data<float>() + static_cast<size_t>(i) * locations;
                    std::vector<float> sorted(begin, begin + locations);
                    std::partial_sort(sorted.begin(), sorted.begin() + top, sorted.end(), std::greater<float>());
                    double sum = 0;
                    for (int j = 0; j < top; ++j) { if (!std::isfinite(sorted[j])) throw std::runtime_error("Invalid SubAD score"); sum += sorted[j]; }
                    imageScores[i] = static_cast<float>(sum / top);
                }
            } else {
                if (config_.variant == PatchCoreVariant::Mahalanobis)
                    patches = compute_->transform(patches, statistics_.mean, statistics_.basis, statistics_.scales, false);
                Tensor distances;
                std::vector<faiss::idx_t> labels(static_cast<size_t>(count) * config_.numNeighbors);
                if (index_->supportsDevice()) {
                    distances = compute_->allocate({count, config_.numNeighbors});
                    auto searched = index_->searchDevice(compute_->device(), count, static_cast<const float*>(patches.deviceData),
                        config_.numNeighbors, static_cast<float*>(distances.deviceData), labels.data());
                    if (!searched) return searched.status();
                } else {
                    // CPU retrieval or older GPU Faiss plugin: explicit host boundary.
                    const auto host = compute_->download(patches);
                    distances.shape.dims = {count, config_.numNeighbors};
                    distances.bytes.resize(labels.size() * sizeof(float));
                    index_->search(count, host.data<float>(), config_.numNeighbors, distances.data<float>(), labels.data());
                }
                if (std::any_of(labels.begin(), labels.end(), [](auto x) { return x < 0; }))
                    return Status::error(ErrorCode::AdapterFailure, "Invalid Faiss labels");
                patchScores = compute_->scores(distances, n, locations, config_.numNeighbors, config_.sqrtDistances, imageScores, maxima);
                if (config_.weightedImageScore && config_.numNeighbors > 1) {
                    // Batch the support queries across images instead of one GPU launch per image.
                    std::vector<float> supports(static_cast<size_t>(n) * searchDimension_);
                    for (int i = 0; i < n; ++i) index_->reconstruct(labels[(static_cast<size_t>(i) * locations + maxima[i]) * config_.numNeighbors], supports.data() + static_cast<size_t>(i) * searchDimension_);
                    std::vector<float> supportDistances(static_cast<size_t>(n) * config_.numNeighbors);
                    std::vector<faiss::idx_t> supportLabels(supportDistances.size());
                    index_->search(n, supports.data(), config_.numNeighbors, supportDistances.data(), supportLabels.data());
                    for (int i = 0; i < n; ++i) {
                        auto* begin = supportDistances.data() + static_cast<size_t>(i) * config_.numNeighbors;
                        for (int j = 0; j < config_.numNeighbors; ++j) {
                            if (!std::isfinite(begin[j]) || begin[j] < 0 || supportLabels[static_cast<size_t>(i)*config_.numNeighbors+j] < 0) throw std::runtime_error("Invalid support distances");
                            if (config_.sqrtDistances) begin[j] = std::sqrt(begin[j]);
                        }
                        const float maximum = *std::max_element(begin, begin + config_.numNeighbors);
                        float denominator = 0;
                        for (int j = 0; j < config_.numNeighbors; ++j) denominator += std::exp(begin[j] - maximum);
                        imageScores[i] *= 1.0F - std::exp(begin[0] - maximum) / denominator;
                    }
                }
            }
            const auto maps = compute_->download(compute_->maps(patchScores, n, h, w, inputSize, config_.gaussianSigma));
            RawPredictionBatch result(n);
            for (int i = 0; i < n; ++i) {
                result[i].score = imageScores[i];
                result[i].anomalyMap = cv::Mat(inputSize.height, inputSize.width, CV_32FC1, const_cast<float*>(maps.data<float>()) + static_cast<size_t>(i)*inputSize.height*inputSize.width, static_cast<size_t>(inputSize.width)*sizeof(float)).clone();
            }
            return result;
        } catch (const std::exception& e) { return Status::error(ErrorCode::CudaFailure, "GPU PatchCore prediction failed", e.what()); }
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs, const cv::Size& inputSize) const {
        if (config_.variant == PatchCoreVariant::SubspaceAD ? statistics_.basis.empty() : !index_)
            return Status::error(ErrorCode::NotInitialized, "PatchCore assets are not loaded");
        if (compute_) return predictGpu(outputs, inputSize);
        auto embedding = aggregatePatchCoreFeatures(outputs, outputBindings_, config_);
        if (!embedding) return embedding.status();
        auto view = feature::viewNchwFloat(embedding.value(), "PatchCore embedding");
        if (!view) return view.status();
        if (view.value().channels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "Aggregated PatchCore embedding dimension does not match manifest",
                                 std::to_string(view.value().channels));
        }

        std::vector<float> queries = feature::flattenPatches(embedding.value());
        const std::size_t locations = static_cast<std::size_t>(view.value().height) * view.value().width;
        const std::size_t queryCount = static_cast<std::size_t>(view.value().batch) * locations;
        if (config_.variant == PatchCoreVariant::SubspaceAD) {
            auto scores = statistics_.residuals(queries);
            if (!scores) return scores.status();
            RawPredictionBatch result(static_cast<std::size_t>(view.value().batch));
            const auto topCount = std::max<std::size_t>(1, static_cast<std::size_t>(
                std::ceil(static_cast<double>(locations) * config_.topScoreFraction)));
            for (std::size_t n = 0; n < result.size(); ++n) {
                const auto* begin = scores.value().data() + n * locations;
                result[n].anomalyMap = feature::makeAnomalyMap(begin, view.value().height,
                    view.value().width, inputSize, config_.gaussianSigma);
                std::vector<float> sorted(begin, begin + locations);
                std::partial_sort(sorted.begin(), sorted.begin() + topCount, sorted.end(), std::greater<float>());
                double sum = 0;
                for (std::size_t i = 0; i < topCount; ++i) sum += sorted[i];
                result[n].score = static_cast<float>(sum / static_cast<double>(topCount));
            }
            return result;
        }
        if (config_.variant == PatchCoreVariant::Mahalanobis) {
            auto whitened = statistics_.transform(queries);
            if (!whitened) return whitened.status();
            queries = std::move(whitened.value());
        }
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
                std::vector<float> support(static_cast<std::size_t>(searchDimension_));
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
    std::shared_ptr<GpuCompute> compute_;
    PatchCoreConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    std::unique_ptr<FaissIndex> index_;
    PatchCoreStatistics statistics_;
    int searchDimension_{0};
};

PatchCoreAdapter::PatchCoreAdapter(PatchCoreConfig config,
                                   std::unordered_map<std::string, std::string> bindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(bindings))) {}
PatchCoreAdapter::~PatchCoreAdapter() = default;
Result<void> PatchCoreAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<SearchExecutionInfo> PatchCoreAdapter::configureSearch(const SearchExecutionConfig& config) {
    return impl_->configureSearch(config);
}
Result<void> PatchCoreAdapter::configureCompute(int device, const std::filesystem::path& directory) { return impl_->configureCompute(device, directory); }
Result<void> PatchCoreAdapter::validateSignature(const TensorSignature& signature) const { return impl_->validateSignature(signature); }
Result<RawPredictionBatch> PatchCoreAdapter::predict(const TensorMap& outputs,
                                                     const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
