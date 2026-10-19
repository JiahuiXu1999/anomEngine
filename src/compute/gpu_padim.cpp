#include "compute/gpu_compute.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

namespace anom::model {
namespace {
void requirePadim(bool condition, const char* message) {
    if (!condition) throw std::invalid_argument(message);
}
int positiveDimension(int64_t value) {
    requirePadim(value > 0 && value <= INT_MAX, "Invalid PaDiM dimension");
    return static_cast<int>(value);
}
size_t floatElements(const Tensor& tensor) {
    auto count = tensor.shape.elementCount();
    requirePadim(tensor.dtype == DataType::Float32 && count.ok() && count.value() > 0 &&
                     count.value() <= SIZE_MAX / sizeof(float) &&
                     tensor.byteSize() == count.value() * sizeof(float),
                 "Invalid PaDiM float tensor");
    return count.value();
}
}  // namespace

Tensor GpuCompute::allocateBytes(size_t bytes) {
    requirePadim(bytes > 0 && bytes <= INT64_MAX, "Invalid device buffer size");
    Tensor result;
    result.dtype = DataType::UInt8;
    result.shape.dims = {static_cast<int64_t>(bytes)};
    auto owner = shared_from_this();
    void* pointer = nullptr;
    checked(api_.allocate(instance_, bytes, &pointer));
    result.deviceData = pointer;
    result.deviceId = device_;
    result.externalByteSize = bytes;
    result.externalOwner = std::shared_ptr<void>(pointer, [owner](void* p) {
        owner->api_.release(owner->instance_, p);
    });
    return result;
}

Result<void> GpuCompute::supportsPadim() const {
    try {
        const uint32_t version = 1;
        run(ANOM_COMPUTE_PADIM_PROBE, version);
        return {};
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::PluginAbiMismatch,
                             "CUDA compute plugin does not support PaDiM", error.what());
    }
}

Tensor GpuCompute::selectFeatures(
    const TensorMap& outputs, const std::unordered_map<std::string, std::string>& bindings,
    const std::vector<std::string>& semantics, const std::vector<int32_t>& indices,
    int height, int width) {
    requirePadim(!semantics.empty() && !indices.empty() && indices.size() <= INT_MAX,
                 "PaDiM requires feature layers and selected channels");
    positiveDimension(height);
    positiveDimension(width);
    std::set<int32_t> unique;
    for (auto index : indices)
        requirePadim(index >= 0 && unique.insert(index).second, "Invalid selected channel index");
    std::set<std::string> tensorNames;
    std::vector<anom_compute_layer> layers;
    int n = 0;
    int64_t totalChannels = 0;
    for (const auto& semantic : semantics) {
        auto binding = bindings.find(semantic);
        requirePadim(binding != bindings.end(), "Missing PaDiM feature binding");
        requirePadim(tensorNames.insert(binding->second).second, "Duplicate PaDiM feature tensor");
        auto found = outputs.find(binding->second);
        requirePadim(found != outputs.end(), "Missing PaDiM feature tensor");
        const auto& tensor = found->second;
        requirePadim(tensor.shape.dims.size() == 4, "PaDiM features must be NCHW");
        floatElements(tensor);
        int batch = positiveDimension(tensor.shape.dims[0]);
        int c = positiveDimension(tensor.shape.dims[1]);
        int h = positiveDimension(tensor.shape.dims[2]);
        int w = positiveDimension(tensor.shape.dims[3]);
        if (layers.empty()) {
            n = batch;
            requirePadim(h == height && w == width, "PaDiM first feature grid differs from statistics");
        } else requirePadim(n == batch, "PaDiM feature batches differ");
        totalChannels += c;
        requirePadim(totalChannels <= INT_MAX, "PaDiM channel count overflow");
        layers.push_back({buffer(tensor), batch, c, h, w});
    }
    requirePadim(*unique.rbegin() < totalChannels, "Selected channel exceeds feature dimension");
    auto result = allocate({n, height, width, static_cast<int64_t>(indices.size())});
    anom_compute_select_features args{layers.data(), layers.size(), indices.data(),
                                      static_cast<int32_t>(indices.size()), height, width, buffer(result)};
    run(ANOM_COMPUTE_SELECT_FEATURES, args);
    return result;
}

Tensor GpuCompute::padimDistances(const Tensor& patches, const Tensor& mean, const Tensor& precision) {
    requirePadim(patches.shape.dims.size() == 4, "PaDiM patches must be NHWD");
    floatElements(patches);
    int n = positiveDimension(patches.shape.dims[0]);
    int h = positiveDimension(patches.shape.dims[1]);
    int w = positiveDimension(patches.shape.dims[2]);
    int d = positiveDimension(patches.shape.dims[3]);
    int locations = positiveDimension(int64_t(h) * w);
    requirePadim(mean.shape.dims == std::vector<int64_t>{locations, d} &&
                     precision.shape.dims == std::vector<int64_t>{locations, d, d},
                 "PaDiM statistics shapes differ");
    floatElements(mean);
    floatElements(precision);
    auto result = allocate({n, h, w});
    anom_compute_padim_distance args{buffer(patches), buffer(mean), buffer(precision),
                                     buffer(result), n, locations, d};
    run(ANOM_COMPUTE_PADIM_DISTANCE, args);
    return result;
}

void GpuCompute::padimAccumulate(const Tensor& patches, int n, int locations, int dimension,
                                 int begin, int count, anom_compute_buffer sums,
                                 anom_compute_buffer moments) {
    floatElements(patches);
    anom_compute_padim_accumulate args{buffer(patches), sums, moments, n, locations,
                                       dimension, begin, count};
    run(ANOM_COMPUTE_PADIM_ACCUMULATE, args);
}

void GpuCompute::padimFinalize(anom_compute_buffer sums, anom_compute_buffer moments,
                               int locations, int dimension, uint64_t samples,
                               double regularization, float* mean, float* precision) {
    const auto l = static_cast<size_t>(positiveDimension(locations));
    const auto d = static_cast<size_t>(positiveDimension(dimension));
    requirePadim(l <= SIZE_MAX / d && l * d <= SIZE_MAX / d / sizeof(double),
                 "PaDiM statistics size overflow");
    anom_compute_padim_finalize args{sums, moments, {mean, l * d * sizeof(float), -1},
                                     {precision, l * d * d * sizeof(float), -1},
                                     locations, dimension, samples, regularization};
    run(ANOM_COMPUTE_PADIM_FINALIZE, args);
}
}  // namespace anom::model
