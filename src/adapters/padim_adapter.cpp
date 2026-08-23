#include "adapters/padim_adapter.h"

#include "adapters/feature_utils.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iterator>
#include <limits>
#include <set>
#include <vector>

namespace anom::model {
namespace {

constexpr std::array<char, 8> kMagic{'A', 'N', 'P', 'A', 'D', 'I', 'M', '\0'};
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
    if (!stream) return Status::error(ErrorCode::IoError, "Unable to open PaDiM artifact", path.string());
    const auto size = stream.tellg();
    if (size <= 0) return Status::error(ErrorCode::ArtifactCorrupt, "PaDiM artifact is empty", path.string());
    std::vector<char> bytes(static_cast<std::size_t>(size));
    stream.seekg(0, std::ios::beg);
    if (!stream.read(bytes.data(), static_cast<std::streamsize>(bytes.size()))) {
        return Status::error(ErrorCode::IoError, "Unable to read complete PaDiM artifact", path.string());
    }
    return bytes;
}

Result<std::size_t> checkedProduct(std::initializer_list<std::size_t> values) {
    std::size_t result = 1;
    for (const auto value : values) {
        if (value != 0 && result > std::numeric_limits<std::size_t>::max() / value) {
            return Status::error(ErrorCode::ArtifactCorrupt, "PaDiM statistics size overflows size_t");
        }
        result *= value;
    }
    return result;
}

Result<std::size_t> checkedAdd(std::size_t left, std::size_t right) {
    if (left > std::numeric_limits<std::size_t>::max() - right) {
        return Status::error(ErrorCode::ArtifactCorrupt,
                             "PaDiM statistics size overflows size_t");
    }
    return left + right;
}

}  // namespace

class PadimAdapter::Impl {
public:
    Impl(PadimConfig config, std::unordered_map<std::string, std::string> bindings)
        : config_(std::move(config)), outputBindings_(std::move(bindings)) {}

    Result<void> loadAssets(const ModelPackage& package) {
        auto statisticsPath = package.resolveArtifact(config_.statisticsFile);
        if (!statisticsPath) return statisticsPath.status();
        auto indicesPath = package.resolveArtifact(config_.channelIndicesFile);
        if (!indicesPath) return indicesPath.status();

        auto indexBytes = readFile(indicesPath.value());
        if (!indexBytes) return indexBytes.status();
        const std::size_t expectedIndexBytes = static_cast<std::size_t>(config_.embeddingDimension) * sizeof(std::int32_t);
        if (indexBytes.value().size() != expectedIndexBytes) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PaDiM channel index artifact has an unexpected size",
                                 std::to_string(indexBytes.value().size()));
        }
        std::vector<std::int32_t> channelIndices(
            static_cast<std::size_t>(config_.embeddingDimension));
        std::memcpy(channelIndices.data(), indexBytes.value().data(), expectedIndexBytes);
        std::set<std::int32_t> uniqueIndices;
        for (const auto index : channelIndices) {
            if (index < 0 || !uniqueIndices.insert(index).second) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "PaDiM channel indices must be unique and non-negative",
                                     std::to_string(index));
            }
        }

        auto statisticsBytes = readFile(statisticsPath.value());
        if (!statisticsBytes) return statisticsBytes.status();
        std::size_t offset = 0;
        std::array<char, 8> magic{};
        if (statisticsBytes.value().size() < magic.size()) {
            return Status::error(ErrorCode::ArtifactCorrupt, "PaDiM statistics header is truncated");
        }
        std::memcpy(magic.data(), statisticsBytes.value().data(), magic.size());
        offset += magic.size();
        std::uint32_t version = 0, height = 0, width = 0, dimension = 0;
        if (magic != kMagic || !readScalar(statisticsBytes.value(), offset, version) ||
            !readScalar(statisticsBytes.value(), offset, height) ||
            !readScalar(statisticsBytes.value(), offset, width) ||
            !readScalar(statisticsBytes.value(), offset, dimension)) {
            return Status::error(ErrorCode::ArtifactCorrupt, "Invalid PaDiM statistics header");
        }
        if (version != kStatisticsVersion || height != static_cast<std::uint32_t>(config_.featureHeight) ||
            width != static_cast<std::uint32_t>(config_.featureWidth) ||
            dimension != static_cast<std::uint32_t>(config_.embeddingDimension)) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PaDiM statistics metadata does not match manifest");
        }
        const std::size_t locations = static_cast<std::size_t>(height) * width;
        auto meanCount = checkedProduct({locations, dimension});
        if (!meanCount) return meanCount.status();
        auto precisionCount = checkedProduct({locations, dimension, dimension});
        if (!precisionCount) return precisionCount.status();
        auto payloadCount = checkedAdd(meanCount.value(), precisionCount.value());
        if (!payloadCount) return payloadCount.status();
        auto totalFloatCount = checkedProduct({payloadCount.value(), sizeof(float)});
        if (!totalFloatCount) return totalFloatCount.status();
        if (statisticsBytes.value().size() - offset != totalFloatCount.value()) {
            return Status::error(ErrorCode::ArtifactCorrupt,
                                 "PaDiM statistics payload size does not match header");
        }
        std::vector<float> mean(meanCount.value());
        std::vector<float> precision(precisionCount.value());
        std::memcpy(mean.data(), statisticsBytes.value().data() + offset,
                    mean.size() * sizeof(float));
        offset += mean.size() * sizeof(float);
        std::memcpy(precision.data(), statisticsBytes.value().data() + offset,
                    precision.size() * sizeof(float));
        for (const auto value : mean) {
            if (!std::isfinite(value)) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "PaDiM mean contains non-finite values");
            }
        }
        const std::size_t matrixSize = static_cast<std::size_t>(dimension) * dimension;
        for (std::size_t location = 0; location < locations; ++location) {
            const float* matrix = precision.data() + location * matrixSize;
            for (std::size_t row = 0; row < dimension; ++row) {
                const float diagonal = matrix[row * dimension + row];
                if (!std::isfinite(diagonal) || diagonal <= 0.0F) {
                    return Status::error(ErrorCode::ArtifactCorrupt,
                                         "PaDiM precision diagonal must be finite and positive",
                                         "location " + std::to_string(location));
                }
                for (std::size_t column = row + 1; column < dimension; ++column) {
                    const float value = matrix[row * dimension + column];
                    const float transpose = matrix[column * dimension + row];
                    if (!std::isfinite(value) || !std::isfinite(transpose)) {
                        return Status::error(ErrorCode::ArtifactCorrupt,
                                             "PaDiM precision contains non-finite values",
                                             "location " + std::to_string(location));
                    }
                    const float tolerance = 1.0e-4F *
                        std::max({1.0F, std::abs(value), std::abs(transpose)});
                    if (std::abs(value - transpose) > tolerance) {
                        return Status::error(ErrorCode::ArtifactCorrupt,
                                             "PaDiM precision matrix must be symmetric",
                                             "location " + std::to_string(location));
                    }
                }
            }
        }
        channelIndices_ = std::move(channelIndices);
        mean_ = std::move(mean);
        precision_ = std::move(precision);
        loaded_ = true;
        return {};
    }

    Result<void> validateSignature(const TensorSignature& signature) const {
        if (!loaded_) {
            return Status::error(ErrorCode::NotInitialized,
                                 "PaDiM statistics are not loaded");
        }
        std::set<std::string> semantics;
        std::set<std::string> tensorNames;
        std::int64_t staticBatch = -1;
        std::int64_t totalChannels = 0;
        bool allChannelsStatic = true;
        bool firstLayer = true;
        for (const auto& semantic : config_.featureLayers) {
            if (!semantics.insert(semantic).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PaDiM feature semantic is duplicated", semantic);
            }
            const auto binding = outputBindings_.find(semantic);
            if (binding == outputBindings_.end()) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PaDiM feature semantic is not bound", semantic);
            }
            if (!tensorNames.insert(binding->second).second) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PaDiM feature layers must bind distinct tensors",
                                     binding->second);
            }
            const auto* spec = signature.findOutput(binding->second);
            if (!spec || spec->dtype != DataType::Float32 || spec->shape.dims.size() != 4) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PaDiM feature must be a float32 NCHW output", binding->second);
            }
            const auto& dims = spec->shape.dims;
            for (const auto dim : dims) {
                if (dim == 0 || dim < -1) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "PaDiM feature has an invalid dimension",
                                         binding->second + " " + spec->shape.toString());
                }
            }
            if (dims[0] > 0) {
                if (staticBatch > 0 && staticBatch != dims[0]) {
                    return Status::error(ErrorCode::TensorSignatureMismatch,
                                         "PaDiM feature tensors have incompatible batch dimensions",
                                         binding->second + " " + spec->shape.toString());
                }
                staticBatch = dims[0];
            }
            if (dims[1] > 0) {
                totalChannels += dims[1];
            } else {
                allChannelsStatic = false;
            }
            if (firstLayer &&
                ((dims[2] > 0 && dims[2] != config_.featureHeight) ||
                 (dims[3] > 0 && dims[3] != config_.featureWidth))) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PaDiM first feature map does not match statistics",
                                     binding->second + " " + spec->shape.toString());
            }
            firstLayer = false;
        }
        if (allChannelsStatic) {
            const auto maximumIndex = *std::max_element(channelIndices_.begin(), channelIndices_.end());
            if (maximumIndex >= totalChannels) {
                return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "PaDiM channel index exceeds aggregated feature channels",
                                     std::to_string(maximumIndex) + " >= " +
                                         std::to_string(totalChannels));
            }
        }
        return {};
    }

    Result<RawPredictionBatch> predict(const TensorMap& outputs, const cv::Size& inputSize) const {
        if (!loaded_) return Status::error(ErrorCode::NotInitialized, "PaDiM statistics are not loaded");
        auto fullEmbedding = feature::aggregateFeaturePyramid(
            outputs, outputBindings_, config_.featureLayers, false, 1, 1, 0);
        if (!fullEmbedding) return fullEmbedding.status();
        auto embedding = feature::selectChannels(fullEmbedding.value(), channelIndices_);
        if (!embedding) return embedding.status();
        auto view = feature::viewNchwFloat(embedding.value(), "PaDiM embedding");
        if (!view) return view.status();
        if (view.value().channels != config_.embeddingDimension ||
            view.value().height != config_.featureHeight || view.value().width != config_.featureWidth) {
            return Status::error(ErrorCode::TensorShapeMismatch,
                                 "PaDiM embedding shape does not match statistics",
                                 embedding.value().shape.toString());
        }

        const std::vector<float> patches = feature::flattenPatches(embedding.value());
        const std::size_t locations = static_cast<std::size_t>(config_.featureHeight) * config_.featureWidth;
        const std::size_t dimension = static_cast<std::size_t>(config_.embeddingDimension);
        std::vector<float> distances(locations);
        std::vector<float> difference(dimension);
        RawPredictionBatch result(static_cast<std::size_t>(view.value().batch));
        for (int n = 0; n < view.value().batch; ++n) {
            for (std::size_t location = 0; location < locations; ++location) {
                const float* patch = patches.data() + (static_cast<std::size_t>(n) * locations + location) * dimension;
                const float* mean = mean_.data() + location * dimension;
                const float* precision = precision_.data() + location * dimension * dimension;
                for (std::size_t d = 0; d < dimension; ++d) difference[d] = patch[d] - mean[d];
                double quadratic = 0.0;
                for (std::size_t row = 0; row < dimension; ++row) {
                    double projected = 0.0;
                    for (std::size_t column = 0; column < dimension; ++column) {
                        projected += static_cast<double>(precision[row * dimension + column]) * difference[column];
                    }
                    quadratic += static_cast<double>(difference[row]) * projected;
                }
                if (!std::isfinite(quadratic) || quadratic < -1.0e-4) {
                    return Status::error(ErrorCode::AdapterFailure,
                                         "PaDiM Mahalanobis quadratic form is invalid",
                                         "location " + std::to_string(location));
                }
                distances[location] = std::sqrt(static_cast<float>(std::max(quadratic, 0.0)));
            }
            auto map = feature::makeAnomalyMap(distances.data(), config_.featureHeight,
                                               config_.featureWidth, inputSize, config_.gaussianSigma);
            double minimum = 0.0, maximum = 0.0;
            cv::minMaxLoc(map, &minimum, &maximum);
            result[static_cast<std::size_t>(n)].score = static_cast<float>(maximum);
            result[static_cast<std::size_t>(n)].anomalyMap = std::move(map);
        }
        return result;
    }

private:
    PadimConfig config_;
    std::unordered_map<std::string, std::string> outputBindings_;
    std::vector<std::int32_t> channelIndices_;
    std::vector<float> mean_;
    std::vector<float> precision_;
    bool loaded_{false};
};

PadimAdapter::PadimAdapter(PadimConfig config,
                           std::unordered_map<std::string, std::string> bindings)
    : impl_(std::make_unique<Impl>(std::move(config), std::move(bindings))) {}
PadimAdapter::~PadimAdapter() = default;
Result<void> PadimAdapter::loadAssets(const ModelPackage& package) { return impl_->loadAssets(package); }
Result<void> PadimAdapter::validateSignature(const TensorSignature& signature) const { return impl_->validateSignature(signature); }
Result<RawPredictionBatch> PadimAdapter::predict(const TensorMap& outputs,
                                                 const cv::Size& modelInputSize) const {
    return impl_->predict(outputs, modelInputSize);
}

}  // namespace anom::model
