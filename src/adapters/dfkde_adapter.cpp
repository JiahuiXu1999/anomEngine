#include "adapters/dfkde_adapter.h"

#include "adapters/feature_utils.h"

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'N', 'D', 'F', 'K', 'D', 'E', '\0'};
constexpr std::uint32_t kStatisticsVersion = 1;

template <typename T>
bool readScalar(const std::vector<char>& data, std::size_t& offset, T& value) {
    if (offset > data.size() || data.size() - offset < sizeof(T)) return false;
    std::memcpy(&value, data.data() + offset, sizeof(T));
    offset += sizeof(T);
    return true;
}

Result<std::vector<char>> readFile(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) return Status::error(ErrorCode::IoError, "Unable to open DFKDE artifact", path.string());
    const auto size = stream.tellg();
    if (size <= 0) return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE artifact is empty", path.string());
    std::vector<char> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        return Status::error(ErrorCode::IoError, "Unable to read complete DFKDE artifact", path.string());
    }
    return bytes;
}

}  // namespace

class DfkdeAdapter::Impl {
public:
    Impl(DFKDEConfig config, std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage& package) {
        auto path = package.resolveArtifact(config_.statisticsFile);
        if (!path) return path.status();
        auto bytes = readFile(path.value());
        if (!bytes) return bytes.status();

        std::size_t offset = 0;
        std::array<char, 8> magic{};
        if (bytes.value().size() < magic.size()) {
            return Status::error(ErrorCode::ArtifactCorrupt, "DFKDE statistics header is truncated");
        }
        std::memcpy(magic.data(), bytes.value().data(), magic.size());
        offset += magic.size();
        std::uint32_t version = 0, dimension = 0, components = 0, galleryCount = 0;
        if (magic != kMagic ||
            !readScalar(bytes.value(), offset, version) ||
            !readScalar(bytes.value(), offset, dimension) ||
            !readScalar(bytes.value(), offset, components) ||
            !readScalar(bytes.value(), offset, galleryCount)) {
            return Status::error(ErrorCode::ArtifactCorrupt, "Invalid DFKDE statistics header");
        }
        if (version != kStatisticsVersion ||
            dimension != static_cast<std::uint32_t>(config_.embeddingDimension)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "DFKDE statistics metadata does not match manifest");
        }
        if (components == 0 || components > dimension || galleryCount == 0) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "DFKDE statistics has invalid dimensions");
        }
        const std::size_t meanCount = dimension;
        const std::size_t eigenCount = static_cast<std::size_t>(dimension) * components;
        const std::size_t gallerySize = static_cast<std::size_t>(galleryCount) * components;
        const std::size_t totalFloats = meanCount + eigenCount + gallerySize;
        if (bytes.value().size() - offset != totalFloats * sizeof(float)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "DFKDE statistics payload size does not match header");
        }
        mean_.assign(totalFloats, 0.0F);
        std::memcpy(mean_.data(), bytes.value().data() + offset, totalFloats * sizeof(float));
        for (const float value : mean_) {
            if (!std::isfinite(value)) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "DFKDE statistics contain non-finite values");
            }
        }
        dimension_ = static_cast<int>(dimension);
        components_ = static_cast<int>(components);
        galleryCount_ = static_cast<int>(galleryCount);
        loaded_ = true;
        return {};
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        std::set<std::string> semantics;
        std::set<std::string> tensorNames;
        std::int64_t staticChannels = 0;
        bool allChannelsStatic = true;
        for (const auto& semantic : config_.featureLayers) {
            if (!semantics.insert(semantic).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature semantic is duplicated", semantic);
            }
            const auto binding = outputBindings_.find(semantic);
            if (binding == outputBindings_.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature semantic is not bound", semantic);
            }
            if (!tensorNames.insert(binding->second).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature layers must bind distinct tensors",
                                     binding->second);
            }
            const auto* spec = signature.findOutput(binding->second);
            if (!spec || spec->dtype != DataType::Float32 || spec->shape.dims.size() != 4) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "DFKDE feature must be a float32 NCHW output", binding->second);
            }
            if (spec->shape.dims[1] > 0) {
                staticChannels += spec->shape.dims[1];
            } else {
                allChannelsStatic = false;
            }
        }
        if (allChannelsStatic && staticChannels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "DFKDE feature channels do not match the embedding dimension",
                                 std::to_string(staticChannels) + " != " +
                                     std::to_string(config_.embeddingDimension));
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs,
                                       const cv::Size& inputSize) const {
        if (!loaded_) return Status::error(ErrorCode::NotInitialized, "DFKDE statistics are not loaded");
        auto embedding = feature::aggregateFeaturePyramid(
            outputs, outputBindings_, config_.featureLayers, false, 1, 1, 0);
        if (!embedding) return embedding.status();
        auto view = feature::viewNchwFloat(embedding.value(), "DFKDE embedding");
        if (!view) return view.status();
        if (view.value().channels != config_.embeddingDimension) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "DFKDE embedding dimension does not match statistics",
                                 embedding.value().shape.toString());
        }

        const std::vector<float> patches = feature::flattenPatches(embedding.value());
        const int locations = view.value().height * view.value().width;
        const int batch = view.value().batch;
        const float inv2s2 = 1.0F / (2.0F * config_.kernelSigma * config_.kernelSigma);

        RawPredictionBatch result(static_cast<std::size_t>(batch));
        std::vector<double> sqDists(static_cast<std::size_t>(galleryCount_));
        std::vector<float> scores(static_cast<std::size_t>(locations));
        for (int n = 0; n < batch; ++n) {
            const float* batchPatches = patches.data() +
                static_cast<std::size_t>(n) * locations * dimension_;
            for (int location = 0; location < locations; ++location) {
                const float* patch = batchPatches + static_cast<std::size_t>(location) * dimension_;
                // Project with the stored PCA basis.
                std::vector<float> projected(static_cast<std::size_t>(components_));
                for (int k = 0; k < components_; ++k) {
                    double sum = 0.0;
                    for (int d = 0; d < dimension_; ++d) {
                        sum += static_cast<double>(patch[d] - mean_[d]) *
                               mean_[static_cast<std::size_t>(dimension_) +
                                     static_cast<std::size_t>(d) * components_ + k];
                    }
                    projected[static_cast<std::size_t>(k)] = static_cast<float>(sum);
                }
                double minSq = std::numeric_limits<double>::infinity();
                for (int g = 0; g < galleryCount_; ++g) {
                    const float* gallery = mean_.data() +
                        static_cast<std::size_t>(dimension_) +
                        static_cast<std::size_t>(dimension_) * components_ +
                        static_cast<std::size_t>(g) * components_;
                    double sq = 0.0;
                    for (int k = 0; k < components_; ++k) {
                        const double diff =
                            static_cast<double>(projected[static_cast<std::size_t>(k)]) - gallery[k];
                        sq += diff * diff;
                    }
                    sqDists[static_cast<std::size_t>(g)] = sq;
                    if (sq < minSq) minSq = sq;
                }
                double logSum = 0.0;
                for (int g = 0; g < galleryCount_; ++g) {
                    logSum += std::exp(-(sqDists[static_cast<std::size_t>(g)] - minSq) * inv2s2);
                }
                scores[static_cast<std::size_t>(location)] =
                    static_cast<float>(minSq * inv2s2 - std::log(logSum));
            }
            auto map = feature::makeAnomalyMap(scores.data(), view.value().height,
                                               view.value().width, inputSize,
                                               config_.gaussianSigma);
            double minimum = 0.0, maximum = 0.0;
            cv::minMaxLoc(map, &minimum, &maximum);
            result[static_cast<std::size_t>(n)].score = static_cast<float>(maximum);
            result[static_cast<std::size_t>(n)].anomalyMap = std::move(map);
        }
        return result;
    }

private:
    DFKDEConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    // Single contiguous buffer: mean [D], eigenvectors [D*K], gallery [N*K].
    std::vector<float> mean_;
    int dimension_{0};
    int components_{0};
    int galleryCount_{0};
    bool loaded_{false};
};

DfkdeAdapter::DfkdeAdapter(DFKDEConfig config,
                           std::unordered_map<std::string, std::string> outputBindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(outputBindings))) {}
DfkdeAdapter::~DfkdeAdapter() = default;
Result<void> DfkdeAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<void> DfkdeAdapter::validateSignature(const TensorSignature& signature) const {
    return impl_->validateSignature(signature);
}
Result<RawPredictionBatch> DfkdeAdapter::predict(const TensorMap& outputs,
                                                 const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
