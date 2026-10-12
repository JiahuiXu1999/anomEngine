#include "pipeline/patchcore_statistics.h"
#include "compute/gpu_compute.h"
#include "adapters/feature_utils.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>

namespace anom::model {
namespace {
constexpr std::array<char, 8> kModelMagic{'A', 'N', 'P', 'C', 'S', 'T', 'A', 'T'};
constexpr std::array<char, 8> kFitMagic{'A', 'N', 'P', 'C', 'S', 'F', 'I', 'T'};
std::FILE* temporaryFile() {
#ifdef _WIN32
    std::FILE* file = nullptr;
    if (tmpfile_s(&file) != 0) return nullptr;
    return file;
#else
    return std::tmpfile();
#endif
}
template <class T>
bool write(std::ostream& s, const T& x) {
    return static_cast<bool>(s.write(reinterpret_cast<const char*>(&x), sizeof(x)));
}
template <class T>
bool read(std::istream& s, T& x) {
    return static_cast<bool>(s.read(reinterpret_cast<char*>(&x), sizeof(x)));
}
bool finite(const cv::Mat& a) {
    for (std::size_t i = 0; i < a.total(); ++i)
        if (!std::isfinite(a.ptr<double>()[i])) return false;
    return true;
}
bool writeMat(std::ostream& s, const cv::Mat& a) {
    return static_cast<bool>(s.write(reinterpret_cast<const char*>(a.data),
                                     static_cast<std::streamsize>(a.total() * sizeof(double))));
}
bool readMat(std::istream& s, cv::Mat& a) {
    return static_cast<bool>(s.read(reinterpret_cast<char*>(a.data),
                                    static_cast<std::streamsize>(a.total() * sizeof(double)))) &&
           finite(a);
}
Status corrupt(const std::filesystem::path& path) {
    return Status::error(ErrorCode::ArtifactCorrupt,
                         "Invalid or incompatible PatchCore statistics/checkpoint", path.string());
}
Result<void> validate(const PatchCoreConfig& c) {
    if ((c.variant != PatchCoreVariant::Mahalanobis && c.variant != PatchCoreVariant::SubspaceAD) ||
        c.embeddingDimension <= 0 || c.pcaComponents < 0 ||
        c.pcaComponents > c.embeddingDimension || !std::isfinite(c.explainedVariance) ||
        c.explainedVariance <= 0 || c.explainedVariance >= 1 ||
        !std::isfinite(c.covarianceShrinkage) || c.covarianceShrinkage < 0 ||
        c.covarianceShrinkage > 1 || !std::isfinite(c.covarianceRegularization) ||
        c.covarianceRegularization <= 0 || c.memoryBankSize <= 0 || c.streamingChunkSize <= 0 ||
        c.memoryBankSize > std::numeric_limits<int>::max() - c.streamingChunkSize)
        return Status::error(ErrorCode::InvalidArgument,
                             "Invalid PatchCore statistical fitting parameters");
    return {};
}
// Deterministic farthest-first merge/reduce; retained vectors are real patches.
std::vector<float> reduceBank(const std::vector<float>& data, int d, std::size_t budget) {
    const auto dimension = static_cast<std::size_t>(d);
    const auto n = data.size() / dimension;
    if (n <= budget) return data;
    std::vector<double> nearest(n, std::numeric_limits<double>::infinity());
    std::vector<bool> used(n, false);
    std::vector<float> result;
    result.reserve(budget * dimension);
    std::size_t next = 0;
    for (std::size_t step = 0; step < budget; ++step) {
        used[next] = true;
        const auto* selected = data.data() + next * dimension;
        result.insert(result.end(), selected, selected + dimension);
        double farthest = -1;
        std::size_t candidate = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (used[i]) continue;
            const auto* x = data.data() + i * dimension;
            double distance = 0;
            for (std::size_t j = 0; j < dimension; ++j) {
                const double delta = static_cast<double>(x[j]) - selected[j];
                distance += delta * delta;
            }
            nearest[i] = std::min(nearest[i], distance);
            if (nearest[i] > farthest) {
                farthest = nearest[i];
                candidate = i;
            }
        }
        next = candidate;
    }
    return result;
}
}  // namespace

Result<Tensor> aggregatePatchCoreFeatures(
    const TensorMap& outputs, const std::unordered_map<std::string, std::string>& bindings,
    const PatchCoreConfig& config) {
    if (!config.meanAggregation)
        return feature::aggregateFeaturePyramid(outputs, bindings, config.featureLayers, true,
                                                config.poolingKernel, config.poolingStride,
                                                config.poolingPadding);
    Tensor result;
    bool first = true;
    for (const auto& semantic : config.featureLayers) {
        const auto binding = bindings.find(semantic);
        if (binding == bindings.end() || !outputs.contains(binding->second))
            return Status::error(ErrorCode::TensorSignatureMismatch,
                                 "PatchCore feature binding is missing", semantic);
        auto pooled = feature::averagePool2d(outputs.at(binding->second), config.poolingKernel,
                                             config.poolingStride, config.poolingPadding);
        if (!pooled) return pooled.status();
        if (first) {
            result = std::move(pooled.value());
            first = false;
        } else {
            if (result.shape.dims != pooled.value().shape.dims)
                return Status::error(ErrorCode::TensorShapeMismatch,
                                     "Mean feature aggregation requires equal shapes");
            auto* dst = result.data<float>();
            const auto* src = pooled.value().data<float>();
            for (std::size_t i = 0; i < result.bytes.size() / sizeof(float); ++i) dst[i] += src[i];
        }
    }
    if (first)
        return Status::error(ErrorCode::InvalidArgument, "PatchCore feature layers are empty");
    const auto divisor = static_cast<float>(config.featureLayers.size());
    for (std::size_t i = 0; i < result.bytes.size() / sizeof(float); ++i)
        result.data<float>()[i] /= divisor;
    return result;
}

Result<void> PatchCoreStatistics::save(const std::filesystem::path& path) const {
    std::ofstream s(path, std::ios::binary | std::ios::trunc);
    const std::uint32_t version = 1, mode = static_cast<std::uint32_t>(variant);
    const std::uint32_t d = static_cast<std::uint32_t>(dimension),
                        r = static_cast<std::uint32_t>(basis.rows);
    if (!write(s, kModelMagic) || !write(s, version) || !write(s, mode) || !write(s, d) ||
        !write(s, r) || !writeMat(s, mean) || !writeMat(s, basis) || !writeMat(s, scales))
        return Status::error(ErrorCode::IoError, "Unable to write PatchCore statistics",
                             path.string());
    return {};
}

Result<PatchCoreStatistics> PatchCoreStatistics::load(const std::filesystem::path& path,
                                                      PatchCoreVariant variant, int dimension) {
    std::ifstream s(path, std::ios::binary | std::ios::ate);
    if (!s)
        return Status::error(ErrorCode::ArtifactMissing, "PatchCore statistics are missing",
                             path.string());
    const auto size = s.tellg();
    s.seekg(0);
    std::array<char, 8> magic{};
    std::uint32_t version = 0, mode = 0, d = 0, r = 0;
    if (!read(s, magic) || !read(s, version) || !read(s, mode) || !read(s, d) || !read(s, r) ||
        magic != kModelMagic || version != 1 || mode != static_cast<std::uint32_t>(variant) ||
        dimension <= 0 || d != static_cast<std::uint32_t>(dimension) || r == 0 || r > d)
        return corrupt(path);
    const std::uint64_t values = static_cast<std::uint64_t>(d) * r + d + r;
    if (values > (std::numeric_limits<std::uint64_t>::max() - 24) / 8 || size < 0 ||
        static_cast<std::uint64_t>(size) != 24 + values * 8)
        return corrupt(path);
    PatchCoreStatistics m;
    m.variant = variant;
    m.dimension = dimension;
    m.mean.create(1, dimension, CV_64F);
    m.basis.create(static_cast<int>(r), dimension, CV_64F);
    m.scales.create(1, static_cast<int>(r), CV_64F);
    if (!readMat(s, m.mean) || !readMat(s, m.basis) || !readMat(s, m.scales)) return corrupt(path);
    for (int i = 0; i < m.basis.rows; ++i) {
        if (m.scales.at<double>(0, i) <= 0 || std::abs(cv::norm(m.basis.row(i)) - 1.0) > 1e-5)
            return corrupt(path);
    }
    return m;
}

Result<std::vector<float>> PatchCoreStatistics::transform(const std::vector<float>& patches) const {
    if (dimension <= 0 || patches.empty() ||
        patches.size() % static_cast<std::size_t>(dimension) != 0)
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "PatchCore transform input dimension differs");
    if (compute) {
        try {
            Tensor input;
            input.shape.dims = {static_cast<int64_t>(patches.size() / dimension), dimension};
            input.setExternalView(patches.data(), patches.size() * sizeof(float));
            const auto host = compute->download(compute->transform(input, mean, basis, scales, false));
            return std::vector<float>(host.data<float>(), host.data<float>() + host.byteSize() / sizeof(float));
        } catch (const std::exception& e) { return Status::error(ErrorCode::CudaFailure, "GPU statistical transform failed", e.what()); }
    }
    const auto n = patches.size() / static_cast<std::size_t>(dimension);
    std::vector<float> result(n * static_cast<std::size_t>(basis.rows));
    // Tile conversion/projection so temporary memory does not grow with the query batch.
    for (std::size_t start = 0; start < n; start += 256) {
        const int rows = static_cast<int>(std::min<std::size_t>(256, n - start));
        cv::Mat input(rows, dimension, CV_32F,
                      const_cast<float*>(patches.data() + start * dimension)),
            centered;
        input.convertTo(centered, CV_64F);
        for (int i = 0; i < rows; ++i) centered.row(i) -= mean;
        cv::Mat projected = centered * basis.t();
        for (int i = 0; i < rows; ++i)
            for (int j = 0; j < basis.rows; ++j) {
                const float value =
                    static_cast<float>(projected.at<double>(i, j) * scales.at<double>(0, j));
                if (!std::isfinite(value))
                    return Status::error(ErrorCode::AdapterFailure, "Non-finite whitened feature");
                result[(start + static_cast<std::size_t>(i)) * basis.rows + j] = value;
            }
    }
    return result;
}

Result<std::vector<float>> PatchCoreStatistics::residuals(const std::vector<float>& patches) const {
    if (dimension <= 0 || patches.empty() ||
        patches.size() % static_cast<std::size_t>(dimension) != 0)
        return Status::error(ErrorCode::TensorShapeMismatch, "SubAD input dimension differs");
    if (compute) {
        try {
            Tensor input;
            input.shape.dims = {static_cast<int64_t>(patches.size() / dimension), dimension};
            input.setExternalView(patches.data(), patches.size() * sizeof(float));
            const auto host = compute->download(compute->transform(input, mean, basis, scales, true));
            return std::vector<float>(host.data<float>(), host.data<float>() + host.byteSize() / sizeof(float));
        } catch (const std::exception& e) { return Status::error(ErrorCode::CudaFailure, "GPU statistical transform failed", e.what()); }
    }
    const auto n = patches.size() / static_cast<std::size_t>(dimension);
    std::vector<float> result(n);
    for (std::size_t start = 0; start < n; start += 256) {
        const int rows = static_cast<int>(std::min<std::size_t>(256, n - start));
        cv::Mat input(rows, dimension, CV_32F,
                      const_cast<float*>(patches.data() + start * dimension)),
            centered;
        input.convertTo(centered, CV_64F);
        for (int i = 0; i < rows; ++i) centered.row(i) -= mean;
        cv::Mat projected = centered * basis.t();
        cv::Mat residual = centered - projected * basis;
        for (int i = 0; i < rows; ++i) {
            const float score = static_cast<float>(residual.row(i).dot(residual.row(i)));
            if (!std::isfinite(score))
                return Status::error(ErrorCode::AdapterFailure, "Non-finite SubAD residual");
            result[start + static_cast<std::size_t>(i)] = score;
        }
    }
    return result;
}

PatchCoreStatisticalFit::PatchCoreStatisticalFit(PatchCoreConfig config, std::shared_ptr<GpuCompute> compute)
    : config_(std::move(config)), compute_(std::move(compute)) {}
PatchCoreStatisticalFit::~PatchCoreStatisticalFit() { clear(); }
void PatchCoreStatisticalFit::clear() noexcept {
    if (spool_) std::fclose(spool_);
    spool_ = nullptr;
    count_ = 0;
    mean_.release();
    m2_.release();
}

Result<void> PatchCoreStatisticalFit::add(const std::vector<float>& patches, std::size_t count) {
    auto valid = validate(config_);
    if (!valid) return valid.status();
    const auto d = static_cast<std::size_t>(config_.embeddingDimension);
    if (count == 0 || count > std::numeric_limits<std::size_t>::max() / d ||
        patches.size() != count * d || count_ > std::numeric_limits<std::size_t>::max() - count ||
        count_ + count > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max()) / d /
                             sizeof(float))
        return Status::error(ErrorCode::InvalidArgument, "Invalid statistical feature batch size");
    for (auto v : patches)
        if (!std::isfinite(v))
            return Status::error(ErrorCode::InvalidArgument, "Non-finite training feature");
    // Work on candidates so failed writes/allocations preserve the fitted statistics.
    cv::Mat nextMean = mean_.clone(), nextM2 = m2_.clone();
    auto nextCount = count_;
    const std::size_t chunkSize = compute_ ? 8192 : 256;
    for (std::size_t start = 0; start < count; start += chunkSize) {
        const int rows = static_cast<int>(std::min<std::size_t>(chunkSize, count - start));
        cv::Mat batchMean, batchM2;
        if (compute_) {
            try { compute_->moments(patches.data() + start * d, rows, config_.embeddingDimension, batchMean, batchM2); }
            catch (const std::exception& e) { return Status::error(ErrorCode::CudaFailure, "GPU covariance failed", e.what()); }
        } else {
            cv::Mat input(rows, config_.embeddingDimension, CV_32F,
                const_cast<float*>(patches.data() + start * d)), batch;
            input.convertTo(batch, CV_64F);
            cv::reduce(batch, batchMean, 0, cv::REDUCE_AVG, CV_64F);
            for (int i = 0; i < rows; ++i) batch.row(i) -= batchMean;
            batchM2 = batch.t() * batch;
        }
        if (nextCount == 0) {
            nextMean = batchMean;
            nextM2 = batchM2;
        } else {
            cv::Mat delta = batchMean - nextMean;
            const double total = static_cast<double>(nextCount) + rows;
            nextM2 +=
                batchM2 + (delta.t() * delta) * (static_cast<double>(nextCount) * rows / total);
            nextMean += delta * (rows / total);
        }
        nextCount += static_cast<std::size_t>(rows);
    }
    if (config_.variant == PatchCoreVariant::Mahalanobis) {
        if (!spool_) spool_ = temporaryFile();
        if (!spool_) return Status::error(ErrorCode::IoError, "Unable to create MH feature cache");
        // Seek to the committed prefix; a failed previous append is overwritten.
        const auto offset = static_cast<std::uint64_t>(count_) * d * sizeof(float);
#ifdef _WIN32
        if (_fseeki64(spool_, static_cast<__int64>(offset), SEEK_SET) != 0)
#else
        if (fseeko(spool_, static_cast<off_t>(offset), SEEK_SET) != 0)
#endif
            return Status::error(ErrorCode::IoError, "Unable to seek MH feature cache");
        if (std::fwrite(patches.data(), sizeof(float), patches.size(), spool_) != patches.size() ||
            std::fflush(spool_) != 0)
            return Status::error(ErrorCode::IoError, "Unable to append MH feature cache");
    }
    mean_ = std::move(nextMean);
    m2_ = std::move(nextM2);
    count_ = nextCount;
    return {};
}

Result<PatchCoreStatistics> PatchCoreStatisticalFit::model() const {
    auto valid = validate(config_);
    if (!valid) return valid.status();
    if (count_ < 2)
        return Status::error(ErrorCode::InvalidArgument,
                             "PCA requires at least two normal patches");
    cv::Mat covariance = m2_ / static_cast<double>(count_ - 1), values, vectors;
    if (compute_) {
        try { compute_->eigen(covariance, values, vectors); }
        catch (const std::exception& e) { return Status::error(ErrorCode::CudaFailure, "GPU eigendecomposition failed", e.what()); }
    } else if (!cv::eigen(covariance, values, vectors))
        return Status::error(ErrorCode::AdapterFailure, "PCA eigendecomposition failed");
    if (!finite(values) || !finite(vectors))
        return Status::error(ErrorCode::AdapterFailure, "PCA eigendecomposition failed");
    int rank = config_.pcaComponents;
    if (rank == 0) {
        if (config_.variant == PatchCoreVariant::Mahalanobis)
            rank = std::min(128, config_.embeddingDimension);
        else {
            double total = 0, accumulated = 0;
            for (int i = 0; i < values.rows; ++i) total += std::max(0.0, values.at<double>(i));
            do {
                accumulated += std::max(0.0, values.at<double>(rank++));
            } while (rank < values.rows && accumulated < config_.explainedVariance * total);
        }
    }
    PatchCoreStatistics m;
    m.compute = compute_;
    m.variant = config_.variant;
    m.dimension = config_.embeddingDimension;
    m.mean = mean_.clone();
    m.basis = vectors.rowRange(0, rank).clone();
    m.scales = cv::Mat::ones(1, rank, CV_64F);
    if (config_.variant == PatchCoreVariant::Mahalanobis) {
        double average = 0;
        for (int i = 0; i < rank; ++i) average += std::max(0.0, values.at<double>(i));
        average /= rank;
        for (int i = 0; i < rank; ++i) {
            const double variance =
                (1.0 - config_.covarianceShrinkage) * std::max(0.0, values.at<double>(i)) +
                config_.covarianceShrinkage * average + config_.covarianceRegularization;
            m.scales.at<double>(0, i) = 1.0 / std::sqrt(variance);
        }
    }
    return m;
}

Result<std::size_t> PatchCoreStatisticalFit::saveArtifacts(
    const std::filesystem::path& indexPath, const std::filesystem::path& statsPath) const {
    auto fitted = model();
    if (!fitted) return fitted.status();
    auto saved = fitted.value().save(statsPath);
    if (!saved) return saved.status();
    if (config_.variant == PatchCoreVariant::SubspaceAD)
        return static_cast<std::size_t>(fitted.value().basis.rows);
    if (!spool_) return Status::error(ErrorCode::NotInitialized, "MH feature cache is unavailable");
    std::rewind(spool_);
    std::vector<float> bank;
    const int rank = fitted.value().basis.rows;
    for (std::size_t start = 0; start < count_;) {
        const auto n = std::min<std::size_t>(config_.streamingChunkSize, count_ - start);
        std::vector<float> raw(n * static_cast<std::size_t>(config_.embeddingDimension));
        if (std::fread(raw.data(), sizeof(float), raw.size(), spool_) != raw.size())
            return Status::error(ErrorCode::IoError, "Unable to replay MH feature cache");
        auto z = fitted.value().transform(raw);
        if (!z) return z.status();
        bank.insert(bank.end(), z.value().begin(), z.value().end());
        if (compute_ && bank.size() / rank > static_cast<size_t>(config_.memoryBankSize)) {
            try {
                Tensor input; input.shape.dims = {static_cast<int64_t>(bank.size()/rank), rank};
                input.setExternalView(bank.data(), bank.size()*sizeof(float));
                auto selected = compute_->coreset(input, config_.memoryBankSize, 0);
                std::vector<float> reduced; reduced.reserve(selected.size()*rank);
                for (auto row : selected) reduced.insert(reduced.end(), bank.data()+row*rank, bank.data()+(row+1)*rank);
                bank = std::move(reduced);
            } catch (const std::exception& e) { return Status::error(ErrorCode::CudaFailure, "GPU MH reduction failed", e.what()); }
        } else bank = reduceBank(bank, rank, static_cast<std::size_t>(config_.memoryBankSize));
        start += n;
    }
    const auto n = bank.size() / static_cast<std::size_t>(rank);
    if (n < static_cast<std::size_t>(config_.numNeighbors))
        return Status::error(ErrorCode::InvalidArgument,
                             "MH memory bank has fewer patches than num_neighbors");
    faiss::IndexFlatL2 index(rank);
    index.add(static_cast<faiss::idx_t>(n), bank.data());
    faiss::write_index(&index, indexPath.string().c_str());
    return n;
}

Result<void> PatchCoreStatisticalFit::saveCheckpoint(const std::filesystem::path& path) const {
    if (count_ == 0)
        return Status::error(ErrorCode::NotInitialized, "No statistical training features");
    std::ofstream s(path, std::ios::binary | std::ios::trunc);
    const std::uint32_t version = 1, mode = static_cast<std::uint32_t>(config_.variant);
    const std::uint32_t d = static_cast<std::uint32_t>(config_.embeddingDimension);
    const std::uint64_t n = count_;
    if (!write(s, kFitMagic) || !write(s, version) || !write(s, mode) || !write(s, d) ||
        !write(s, n) || !writeMat(s, mean_) || !writeMat(s, m2_))
        return Status::error(ErrorCode::IoError, "Unable to write statistical checkpoint",
                             path.string());
    if (config_.variant == PatchCoreVariant::Mahalanobis) {
        std::rewind(spool_);
        std::array<float, 4096> buffer{};
        for (std::uint64_t left = n * d; left > 0;) {
            const auto take =
                static_cast<std::size_t>(std::min<std::uint64_t>(left, buffer.size()));
            if (std::fread(buffer.data(), sizeof(float), take, spool_) != take ||
                !s.write(reinterpret_cast<const char*>(buffer.data()),
                         static_cast<std::streamsize>(take * sizeof(float))))
                return Status::error(ErrorCode::IoError, "Unable to copy MH checkpoint features",
                                     path.string());
            left -= take;
        }
    }
    return {};
}

Result<void> PatchCoreStatisticalFit::loadCheckpoint(const std::filesystem::path& path) {
    auto valid = validate(config_);
    if (!valid) return valid.status();
    std::ifstream s(path, std::ios::binary | std::ios::ate);
    if (!s)
        return Status::error(ErrorCode::IoError, "Unable to open statistical checkpoint",
                             path.string());
    const auto size = s.tellg();
    s.seekg(0);
    std::array<char, 8> magic{};
    std::uint32_t version = 0, mode = 0, d = 0;
    std::uint64_t n = 0;
    if (!read(s, magic) || !read(s, version) || !read(s, mode) || !read(s, d) || !read(s, n) ||
        magic != kFitMagic || version != 1 || mode != static_cast<std::uint32_t>(config_.variant) ||
        d != static_cast<std::uint32_t>(config_.embeddingDimension) || n == 0 ||
        n > std::numeric_limits<std::size_t>::max())
        return corrupt(path);
    const std::uint64_t matrixValues = static_cast<std::uint64_t>(d) * d + d;
    if (matrixValues > (std::numeric_limits<std::uint64_t>::max() - 28) / sizeof(double))
        return corrupt(path);
    const std::uint64_t matrixBytes = matrixValues * sizeof(double);
    if (size < 0 || static_cast<std::uint64_t>(size) < 28 + matrixBytes) return corrupt(path);
    const auto remaining = static_cast<std::uint64_t>(size) - 28 - matrixBytes;
    if (config_.variant == PatchCoreVariant::Mahalanobis) {
        if (remaining % sizeof(float) != 0 || remaining / sizeof(float) / d != n ||
            remaining / sizeof(float) % d != 0)
            return corrupt(path);
    } else if (remaining != 0)
        return corrupt(path);
    PatchCoreStatisticalFit candidate(config_);
    candidate.mean_.create(1, config_.embeddingDimension, CV_64F);
    candidate.m2_.create(config_.embeddingDimension, config_.embeddingDimension, CV_64F);
    if (!readMat(s, candidate.mean_) || !readMat(s, candidate.m2_)) return corrupt(path);
    if (config_.variant == PatchCoreVariant::Mahalanobis) {
        candidate.spool_ = temporaryFile();
        if (!candidate.spool_)
            return Status::error(ErrorCode::IoError, "Unable to restore MH feature cache");
        std::array<float, 4096> buffer{};
        for (std::uint64_t left = remaining / sizeof(float); left > 0;) {
            const auto take =
                static_cast<std::size_t>(std::min<std::uint64_t>(left, buffer.size()));
            if (!s.read(reinterpret_cast<char*>(buffer.data()),
                        static_cast<std::streamsize>(take * sizeof(float))))
                return corrupt(path);
            for (std::size_t i = 0; i < take; ++i)
                if (!std::isfinite(buffer[i])) return corrupt(path);
            if (std::fwrite(buffer.data(), sizeof(float), take, candidate.spool_) != take)
                return Status::error(ErrorCode::IoError, "Unable to restore MH feature cache");
            left -= take;
        }
    }
    clear();
    count_ = static_cast<std::size_t>(n);
    mean_ = std::move(candidate.mean_);
    m2_ = std::move(candidate.m2_);
    std::swap(spool_, candidate.spool_);
    return {};
}
}  // namespace anom::model
