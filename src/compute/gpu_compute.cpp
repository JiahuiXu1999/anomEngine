#include "compute/gpu_compute.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/imgproc.hpp>
#include <set>
#include <stdexcept>
namespace anom::model {
namespace {
anom_compute_buffer host(const void* p, size_t n) { return {const_cast<void*>(p), n, -1}; }
void require(bool b, const char* message) {
    if (!b) throw std::invalid_argument(message);
}
int integer(int64_t v) {
    require(v > 0 && v <= INT_MAX, "Invalid GPU tensor dimension");
    return int(v);
}
}  // namespace
Result<std::shared_ptr<GpuCompute>> GpuCompute::load(int device, const std::filesystem::path& directory) {
    if (device < 0) return Status::error(ErrorCode::InvalidArgument, "Negative compute device");
    auto lib = plugins::DynamicLibrary::open(plugins::pluginPath(
        directory.empty() ? plugins::defaultPluginDirectory() : directory, "compute", "cuda"));
    if (!lib)
        return Status::error(ErrorCode::PluginNotFound, "CUDA compute plugin unavailable",
                             lib.status().describe());
    auto c = std::shared_ptr<GpuCompute>(new GpuCompute);
    c->library_ = std::move(lib.value());
    c->device_ = device;
    auto query = reinterpret_cast<anom_compute_query_v1_fn>(c->library_->symbol("anom_compute_query_v1"));
    c->api_.struct_size = sizeof(c->api_);
    if (!query || query(1, &c->api_) || c->api_.struct_size < sizeof(c->api_) || c->api_.abi_version != 1 ||
        !c->api_.create || !c->api_.destroy || !c->api_.allocate || !c->api_.release || !c->api_.copy ||
        !c->api_.run || !c->api_.error)
        return Status::error(ErrorCode::PluginAbiMismatch, "Invalid CUDA compute ABI");
    try {
        c->checked(c->api_.create(device, &c->instance_));
        if (!c->instance_) throw std::runtime_error("Null compute instance");
    } catch (const std::exception& e) {
        return Status::error(ErrorCode::DeviceUnavailable, "CUDA compute initialization failed", e.what());
    }
    return c;
}
GpuCompute::~GpuCompute() {
    if (instance_) api_.destroy(instance_);
}
Result<void> GpuCompute::supportsOperations(uint64_t operations) const {
    if (!operations)
        return Status::error(ErrorCode::InvalidArgument, "No CUDA operations requested");
    const anom_compute_capabilities capabilities{operations};
    const int32_t status = api_.run(instance_, ANOM_COMPUTE_CAPABILITIES,
                                    &capabilities, sizeof(capabilities));
    try {
        checked(status);
        return {};
    } catch (const std::exception& error) {
        // A legacy compute plugin rejects the new operation with InvalidArgument.
        // Treat that as a capability mismatch so AUTO can select the CPU path.
        const auto code = status == static_cast<int32_t>(ErrorCode::InvalidArgument)
            ? ErrorCode::PluginAbiMismatch
            : (status >= static_cast<int32_t>(ErrorCode::InvalidArgument) &&
                       status <= static_cast<int32_t>(ErrorCode::DeviceUnavailable)
                   ? static_cast<ErrorCode>(status) : ErrorCode::PluginAbiMismatch);
        return Status::error(code, "Required CUDA compute operations are unavailable", error.what());
    }
}
void GpuCompute::checked(int32_t s) const {
    if (!s) return;
    std::string message(api_.error(nullptr, 0), '\0');
    if (!message.empty()) {
        api_.error(message.data(), message.size());
        message.pop_back();
    }
    throw std::runtime_error("CUDA compute: " + message);
}
anom_compute_buffer GpuCompute::buffer(const Tensor& t) {
    return t.deviceData ? anom_compute_buffer{t.deviceData, t.byteSize(), t.deviceId}
                        : host(t.data<std::byte>(), t.byteSize());
}
Tensor GpuCompute::allocate(std::vector<int64_t> shape) {
    Tensor t;
    t.shape.dims = std::move(shape);
    auto count = t.shape.elementCount();
    if (!count) throw std::invalid_argument(count.status().describe());
    require(count.value() && count.value() <= SIZE_MAX / 4, "Invalid device tensor size");
    auto owner = shared_from_this();
    void* p = nullptr;
    checked(api_.allocate(instance_, count.value() * 4, &p));
    t.deviceData = p;
    t.deviceId = device_;
    t.externalByteSize = count.value() * 4;
    t.externalOwner =
        std::shared_ptr<void>(p, [owner](void* ptr) { owner->api_.release(owner->instance_, ptr); });
    return t;
}
Tensor GpuCompute::upload(const Tensor& t) {
    require(t.dtype == DataType::Float32, "Compute requires float32");
    if (t.deviceData) {
        require(t.deviceId == device_, "Device mismatch");
        return t;
    }
    auto out = allocate(t.shape.dims);
    require(out.byteSize() == t.byteSize(), "Tensor byte count mismatch");
    checked(api_.copy(instance_, buffer(out), buffer(t)));
    return out;
}
Tensor GpuCompute::download(const Tensor& t) const {
    if (!t.deviceData) return t;
    Tensor out;
    out.dtype = t.dtype;
    out.shape = t.shape;
    out.bytes.resize(t.byteSize());
    checked(api_.copy(instance_, buffer(out), buffer(t)));
    return out;
}
Tensor GpuCompute::aggregate(const TensorMap& outputs,
                             const std::unordered_map<std::string, std::string>& bindings,
                             const PatchCoreConfig& config) {
    require(config.embeddingDimension > 0, "Invalid embedding dimension");
    return aggregateImpl(outputs, bindings, config.featureLayers, config.meanAggregation,
                         config.poolingKernel, config.poolingStride, config.poolingPadding,
                         config.embeddingDimension);
}
Tensor GpuCompute::aggregateFeatures(const TensorMap& outputs,
                                    const std::unordered_map<std::string, std::string>& bindings,
                                    const std::vector<std::string>& semantics) {
    return aggregateImpl(outputs, bindings, semantics, false, 1, 1, 0, 0);
}
Tensor GpuCompute::aggregateImpl(const TensorMap& outputs,
                                const std::unordered_map<std::string, std::string>& bindings,
                                const std::vector<std::string>& semantics, bool mean,
                                int kernel, int stride, int padding, int expectedDimension) {
    std::vector<anom_compute_layer> layers;
    std::set<std::string> names;
    int dimension = 0, batch = 0, h = 0, w = 0;
    require(stride > 0 && kernel > 0 && kernel <= 1024 && padding >= 0 && padding <= 1024,
            "Invalid pooling");
    for (const auto& semantic : semantics) {
        auto b = bindings.find(semantic);
        require(b != bindings.end(), "Missing feature binding");
        require(names.insert(b->second).second, "Duplicate feature tensor binding");
        auto found = outputs.find(b->second);
        require(found != outputs.end(), "Missing feature tensor");
        const auto& t = found->second;
        require(t.dtype == DataType::Float32 && t.shape.dims.size() == 4, "Features must be float32 NCHW");
        int n = integer(t.shape.dims[0]), c = integer(t.shape.dims[1]), ih = integer(t.shape.dims[2]),
            iw = integer(t.shape.dims[3]);
        auto size = t.shape.elementCount();
        require(size.ok() && size.value() <= SIZE_MAX / 4 && size.value() * 4 == t.byteSize(),
                "Feature size mismatch");
        require(c <= INT_MAX - dimension, "Channel overflow");
        require(int64_t(ih) + 2LL * padding >= kernel && int64_t(iw) + 2LL * padding >= kernel,
                "Empty pooled feature grid");
        require(layers.empty() || n == batch, "Feature batches differ");
        dimension += c;
        layers.push_back({buffer(t), n, c, ih, iw});
        if (layers.size() == 1) {
            batch = n;
            h = integer((int64_t(ih) + 2LL * padding - kernel) / stride + 1);
            w = integer((int64_t(iw) + 2LL * padding - kernel) / stride + 1);
        }
    }
    require(!layers.empty(), "No feature layers");
    if (mean) dimension = layers.front().c;
    require(expectedDimension == 0 || dimension == expectedDimension, "Embedding dimension mismatch");
    auto out = allocate({batch, h, w, dimension});
    anom_compute_aggregate a{layers.data(), layers.size(), kernel, stride, padding, mean ? 1 : 0,
                            buffer(out)};
    run(ANOM_COMPUTE_AGGREGATE, a);
    return out;
}
Tensor GpuCompute::project(const std::vector<float>& x, int n, int d, const std::vector<float>& r, int k) {
    require(x.size() == size_t(n) * d && r.size() == size_t(d) * k, "Projection shape mismatch");
    auto out = allocate({n, k});
    anom_compute_project a{host(x.data(), x.size() * 4), host(r.data(), r.size() * 4), buffer(out), n, d, k};
    run(ANOM_COMPUTE_PROJECT, a);
    return out;
}
std::vector<int64_t> GpuCompute::coreset(const Tensor& x, int k, int first) const {
    require(x.shape.dims.size() == 2, "Coreset matrix must have rank two");
    require(k > 0 && k <= x.shape.dims[0] && first >= 0 && first < x.shape.dims[0],
            "Invalid coreset count/seed");
    std::vector<int64_t> selected(k);
    anom_compute_coreset a{buffer(x), integer(x.shape.dims[0]), integer(x.shape.dims[1]), k,
                           first,     selected.data()};
    run(ANOM_COMPUTE_CORESET, a);
    return selected;
}
Tensor GpuCompute::transform(const Tensor& x, const cv::Mat& mean, const cv::Mat& basis,
                             const cv::Mat& scales, bool residual) {
    require(mean.type() == CV_64F && basis.type() == CV_64F && mean.rows == 1 && mean.cols == basis.cols &&
                mean.isContinuous() && basis.isContinuous(),
            "Invalid PCA statistics");
    require(residual ||
                (scales.type() == CV_64F && scales.total() == size_t(basis.rows) && scales.isContinuous()),
            "Invalid whitening scales");
    int d = basis.cols;
    require(d > 0 && x.byteSize() % (size_t(d) * 4) == 0, "Transform input mismatch");
    int rows = integer(int64_t(x.byteSize() / (size_t(d) * 4)));
    auto out = allocate({rows, residual ? 1 : basis.rows});
    anom_compute_transform a{buffer(x),
                             buffer(out),
                             rows,
                             d,
                             basis.rows,
                             residual ? 1 : 0,
                             mean.ptr<double>(),
                             basis.ptr<double>(),
                             scales.empty() ? nullptr : scales.ptr<double>()};
    run(ANOM_COMPUTE_TRANSFORM, a);
    return out;
}
void GpuCompute::moments(const float* x, int n, int d, cv::Mat& mean, cv::Mat& m2) const {
    mean.create(1, d, CV_64F);
    m2.create(d, d, CV_64F);
    anom_compute_moments a{host(x, size_t(n) * d * 4), n, d, mean.ptr<double>(), m2.ptr<double>()};
    run(ANOM_COMPUTE_MOMENTS, a);
}
void GpuCompute::eigen(const cv::Mat& covariance, cv::Mat& values, cv::Mat& vectors) const {
    require(covariance.type() == CV_64F && covariance.rows == covariance.cols && covariance.isContinuous(),
            "Invalid covariance");
    int d = covariance.rows;
    values.create(d, 1, CV_64F);
    vectors.create(d, d, CV_64F);
    anom_compute_eigen a{d, covariance.ptr<double>(), values.ptr<double>(), vectors.ptr<double>()};
    run(ANOM_COMPUTE_EIGEN, a);
}
Result<PreprocessedBatch> GpuCompute::processImages(const std::vector<cv::Mat>& images,
                                                    const InputConfig& config) {
    try {
        return preprocess(images, config);
    } catch (const std::invalid_argument& e) {
        return Status::error(ErrorCode::InvalidImage, "Invalid GPU input image", e.what());
    } catch (const std::exception& e) {
        return Status::error(ErrorCode::CudaFailure, "GPU preprocessing failed", e.what());
    }
}
PreprocessedBatch GpuCompute::preprocess(const std::vector<cv::Mat>& images, const InputConfig& config) {
    require(!images.empty() && images.size() <= INT_MAX, "Invalid image batch");
    cv::Size crop = config.centerCrop.value_or(cv::Size(config.resizeWidth, config.resizeHeight));
    std::vector<anom_compute_image> views;
    std::vector<cv::Mat> resized;
    resized.reserve(images.size());
    PreprocessedBatch result;
    int interpolation = config.interpolation;
    bool cpuResize = interpolation != cv::INTER_LINEAR && interpolation != cv::INTER_NEAREST;
    for (const auto& image : images) {
        require(!image.empty() && image.depth() == CV_8U &&
                    (image.channels() == 1 || image.channels() == 3 || image.channels() == 4),
                "Expected 8-bit 1/3/4-channel image");
        const cv::Mat* source = &image;
        if (cpuResize) {
            resized.emplace_back();
            cv::resize(image, resized.back(), {config.resizeWidth, config.resizeHeight}, 0, 0, interpolation);
            source = &resized.back();
        }
        views.push_back({source->data, source->step, source->rows, source->cols, source->channels()});
        result.geometry.push_back({image.size(),
                                   {config.resizeWidth, config.resizeHeight},
                                   {(config.resizeWidth - crop.width) / 2,
                                    (config.resizeHeight - crop.height) / 2, crop.width, crop.height}});
    }
    result.tensor = allocate(config.layout == TensorLayout::NCHW
                                 ? std::vector<int64_t>{int64_t(images.size()), 3, crop.height, crop.width}
                                 : std::vector<int64_t>{int64_t(images.size()), crop.height, crop.width, 3});
    anom_compute_preprocess a{};
    a.images = views.data();
    a.count = views.size();
    a.resize_h = config.resizeHeight;
    a.resize_w = config.resizeWidth;
    a.crop_h = crop.height;
    a.crop_w = crop.width;
    a.rgb = config.color == "RGB";
    a.nchw = config.layout == TensorLayout::NCHW;
    a.interpolation = cpuResize ? 0 : interpolation;
    std::copy(config.mean.begin(), config.mean.end(), a.mean);
    std::copy(config.std.begin(), config.std.end(), a.stddev);
    a.output = buffer(result.tensor);
    run(ANOM_COMPUTE_PREPROCESS, a);
    return result;
}
Tensor GpuCompute::maps(const Tensor& scores, int n, int h, int w, cv::Size size, float sigma) {
    require(std::isfinite(sigma) && sigma >= 0 && sigma <= 511, "Invalid Gaussian sigma");
    int k = sigma > 0 ? (cvRound(double(sigma) * 8 + 1) | 1) : 1;
    cv::Mat weights = sigma > 0 ? cv::getGaussianKernel(k, sigma, CV_32F) : cv::Mat::ones(1, 1, CV_32F);
    auto out = allocate({n, size.height, size.width});
    anom_compute_map a{buffer(scores), buffer(out),          n, h, w, size.height,
                       size.width,     weights.ptr<float>(), k};
    run(ANOM_COMPUTE_MAP, a);
    return out;
}
Tensor GpuCompute::scores(const Tensor& distances, int n, int locations, int neighbors, bool root,
                          std::vector<float>& maxima, std::vector<int64_t>& indices) {
    require(n > 0 && locations > 0 && neighbors > 0, "Invalid score dimensions");
    const auto count = size_t(n) * locations;
    require(count <= SIZE_MAX / size_t(neighbors) / sizeof(float) &&
                distances.byteSize() == count * neighbors * sizeof(float),
            "Score buffer size mismatch");
    maxima.assign(n, -std::numeric_limits<float>::infinity());
    indices.assign(n, 0);
    if (!distances.deviceData) {
        // Measurements show that uploading a small distance matrix just for a
        // reduction is slower. CPU retrieval keeps this reduction on the host.
        Tensor out;
        out.shape.dims = {n, locations};
        out.bytes.resize(count * sizeof(float));
        const float* values = distances.data<float>();
        for (size_t i = 0; i < count; ++i) {
            for (int k = 0; k < neighbors; ++k)
                require(std::isfinite(values[i * neighbors + k]) && values[i * neighbors + k] >= 0,
                        "Invalid neighbor distance");
            const float v = root ? std::sqrt(values[i * neighbors]) : values[i * neighbors];
            out.data<float>()[i] = v;
            const size_t image = i / locations;
            if (v > maxima[image]) {
                maxima[image] = v;
                indices[image] = int64_t(i % locations);
            }
        }
        return out;
    }
    auto out = allocate({n, locations});
    anom_compute_scores a{buffer(distances), buffer(out),   n, locations, neighbors, root ? 1 : 0,
                          maxima.data(),     indices.data()};
    run(ANOM_COMPUTE_SCORES, a);
    return out;
}
}  // namespace anom::model
