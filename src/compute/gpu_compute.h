#pragma once
#include "model/manifest.h"
#include "model/types.h"
#include "plugin_api/anom_compute_plugin.h"
#include "plugins/dynamic_library.h"
namespace anom::model {
class GpuCompute : public std::enable_shared_from_this<GpuCompute> {
   public:
    static Result<std::shared_ptr<GpuCompute>> load(int device, const std::filesystem::path& directory = {});
    ~GpuCompute();
    GpuCompute(const GpuCompute&) = delete;
    int device() const { return device_; }
    Tensor allocate(std::vector<int64_t> shape);
    Tensor upload(const Tensor& tensor);
    Tensor download(const Tensor& tensor) const;
    static anom_compute_buffer buffer(const Tensor& tensor);
    template <class T>
    void run(int op, const T& args) const {
        checked(api_.run(instance_, op, &args, sizeof(args)));
    }
    Tensor aggregate(const TensorMap&, const std::unordered_map<std::string, std::string>&,
                     const PatchCoreConfig&);
    Tensor project(const std::vector<float>& x, int rows, int dimension, const std::vector<float>& matrix,
                   int rank);
    std::vector<int64_t> coreset(const Tensor& features, int count, int first) const;
    Tensor transform(const Tensor& input, const cv::Mat& mean, const cv::Mat& basis, const cv::Mat& scales,
                     bool residual);
    void moments(const float* x, int rows, int dimension, cv::Mat& mean, cv::Mat& m2) const;
    void eigen(const cv::Mat& covariance, cv::Mat& values, cv::Mat& vectors) const;
    Result<PreprocessedBatch> processImages(const std::vector<cv::Mat>& images, const InputConfig& config);
    PreprocessedBatch preprocess(const std::vector<cv::Mat>& images, const InputConfig& config);
    Tensor maps(const Tensor& scores, int batch, int h, int w, cv::Size output, float sigma);
    Tensor scores(const Tensor& distances, int batch, int locations, int neighbors, bool sqrt,
                  std::vector<float>& maxima, std::vector<int64_t>& indices);

   private:
    GpuCompute() = default;
    void checked(int32_t status) const;
    std::shared_ptr<plugins::DynamicLibrary> library_;
    anom_compute_api_v1 api_{};
    void* instance_{};
    int device_{};
};
}  // namespace anom::model
