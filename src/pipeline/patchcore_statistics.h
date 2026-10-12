#pragma once

#include "model/manifest.h"
#include <cstdio>
#include <memory>

namespace anom::model {
class GpuCompute;

Result<Tensor> aggregatePatchCoreFeatures(
    const TensorMap& outputs, const std::unordered_map<std::string, std::string>& bindings,
    const PatchCoreConfig& config);

// Frozen PCA model. Rows of basis are orthonormal; scales implement regularized
// covariance whitening for MH. SubAD scores the residual in the original space.
struct PatchCoreStatistics {
    PatchCoreVariant variant{PatchCoreVariant::Original};
    int dimension{0};
    cv::Mat mean;    // 1 x D, double
    cv::Mat basis;   // R x D, double
    std::shared_ptr<GpuCompute> compute;
    cv::Mat scales;  // 1 x R, double

    Result<void> save(const std::filesystem::path& path) const;
    static Result<PatchCoreStatistics> load(const std::filesystem::path& path,
                                            PatchCoreVariant variant, int dimension);
    Result<std::vector<float>> transform(const std::vector<float>& patches) const;
    Result<std::vector<float>> residuals(const std::vector<float>& patches) const;
};

// Full-covariance PCA with bounded RAM. MH caches raw features in an owned
// temporary file, then replays them after freezing PCA/whitening. SubAD only
// retains sufficient statistics. This is not the paper's IPCA implementation.
class PatchCoreStatisticalFit {
   public:
    explicit PatchCoreStatisticalFit(PatchCoreConfig config, std::shared_ptr<GpuCompute> compute = {});
    ~PatchCoreStatisticalFit();
    PatchCoreStatisticalFit(const PatchCoreStatisticalFit&) = delete;
    PatchCoreStatisticalFit& operator=(const PatchCoreStatisticalFit&) = delete;
    Result<void> add(const std::vector<float>& patches, std::size_t count);
    Result<std::size_t> saveArtifacts(const std::filesystem::path& index,
                                      const std::filesystem::path& statistics) const;
    Result<void> saveCheckpoint(const std::filesystem::path& path) const;
    Result<void> loadCheckpoint(const std::filesystem::path& path);
    void clear() noexcept;
    std::size_t count() const noexcept { return count_; }

   private:
    Result<PatchCoreStatistics> model() const;
    PatchCoreConfig config_;
    std::shared_ptr<GpuCompute> compute_;
    std::size_t count_{0};
    cv::Mat mean_, m2_;
    mutable std::FILE* spool_{nullptr};
};

}  // namespace anom::model
