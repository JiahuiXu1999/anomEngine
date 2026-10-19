#include "compute/gpu_compute.h"

#include <climits>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace anom::model {
namespace {
void requireDfkde(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
int dfkdeDimension(int64_t value) {
    requireDfkde(value > 0 && value <= INT_MAX, "Invalid DFKDE dimension");
    return static_cast<int>(value);
}
size_t dfkdeFloatElements(const Tensor& tensor) {
    const auto count = tensor.shape.elementCount();
    requireDfkde(tensor.dtype == DataType::Float32 && count.ok() && count.value() > 0 &&
                     count.value() <= SIZE_MAX / sizeof(float) &&
                     tensor.byteSize() == count.value() * sizeof(float),
                 "Invalid DFKDE float tensor");
    return count.value();
}
} // namespace

Tensor GpuCompute::dfkdeScores(const Tensor& patches, const Tensor& mean, const Tensor& basis,
                               const Tensor& gallery, float sigma, size_t workspaceBytes) {
    requireDfkde(patches.shape.dims.size() == 2 || patches.shape.dims.size() == 4,
                 "DFKDE patches must be rows by dimension or NHWD");
    const size_t patchElements = dfkdeFloatElements(patches);
    for (auto value : patches.shape.dims)
        dfkdeDimension(value);
    const int dimension = dfkdeDimension(patches.shape.dims.back());
    const size_t rowCount = patchElements / dimension;
    requireDfkde(rowCount <= INT_MAX, "DFKDE patch row count overflow");
    requireDfkde(mean.shape.dims == std::vector<int64_t>{dimension} && basis.shape.dims.size() == 2 &&
                     basis.shape.dims[0] == dimension && gallery.shape.dims.size() == 2,
                 "DFKDE statistics shapes differ");
    const int rank = dfkdeDimension(basis.shape.dims[1]);
    const int galleryCount = dfkdeDimension(gallery.shape.dims[0]);
    requireDfkde(rank <= dimension && gallery.shape.dims[1] == rank,
                 "Invalid DFKDE PCA rank or gallery shape");
    dfkdeFloatElements(mean);
    dfkdeFloatElements(basis);
    dfkdeFloatElements(gallery);
    requireDfkde(std::isfinite(sigma) && sigma > 0.0F && std::isfinite(1.0F / (2.0F * sigma * sigma)),
                 "DFKDE sigma yields a non-finite inverse bandwidth");
    requireDfkde(workspaceBytes >= size_t(rank) * sizeof(float) + 2 * sizeof(double) + sizeof(int),
                 "DFKDE workspace cannot hold one projected patch and reduction");
    auto shape = patches.shape.dims;
    shape.pop_back();
    auto result = allocate(std::move(shape));
    const anom_compute_dfkde args{buffer(patches),
                                  buffer(mean),
                                  buffer(basis),
                                  buffer(gallery),
                                  buffer(result),
                                  static_cast<int>(rowCount),
                                  dimension,
                                  rank,
                                  galleryCount,
                                  sigma,
                                  static_cast<uint64_t>(workspaceBytes)};
    run(ANOM_COMPUTE_DFKDE, args);
    return result;
}
} // namespace anom::model
