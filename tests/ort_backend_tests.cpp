#include "backends/backend_factory.h"
#include "model/inference_session.h"

#include <cmath>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace anom::model;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

Tensor makeInput() {
    Tensor tensor;
    tensor.dtype = DataType::Float32;
    tensor.shape.dims = {2, 3, 2, 2};
    const std::size_t count = 24;
    tensor.bytes.resize(count * sizeof(float));
    for (std::size_t i = 0; i < count; ++i) {
        tensor.data<float>()[i] = static_cast<float>(i) * 0.25F;
    }
    return tensor;
}

void testIdentityModel() {
    BackendConfig config;
    config.backend = RuntimeBackend::OnnxRuntime;
    config.onnxPath = std::filesystem::path(ANOM_TEST_DATA_DIR) / "identity_dynamic.onnx";
    config.maxBatchSize = 8;
    config.intraOpThreads = 1;
    config.interOpThreads = 1;

    auto backend = createRuntimeBackend(config.backend);
    require(backend.ok(), backend.status().describe());
    auto loaded = backend.value()->load(config);
    require(loaded.ok(), loaded.status().describe());

    const auto& signature = backend.value()->signature();
    require(signature.inputs.size() == 1, "Identity model input count is wrong");
    require(signature.outputs.size() == 1, "Identity model output count is wrong");
    require(signature.inputs.front().name == "input", "Identity input name is wrong");
    require(signature.outputs.front().name == "output", "Identity output name is wrong");
    require(signature.inputs.front().shape.dims ==
                std::vector<std::int64_t>({-1, 3, -1, -1}),
            "Dynamic input signature was not preserved");
    require(backend.value()->maxBatchSize() == 8,
            "Configured dynamic maximum batch size was not reported");

    Tensor input = makeInput();
    TensorMap inputs;
    inputs.emplace("input", input);
    auto outputs = backend.value()->infer(inputs);
    require(outputs.ok(), outputs.status().describe());
    const auto found = outputs.value().find("output");
    require(found != outputs.value().end(), "Identity output is missing");
    require(found->second.shape.dims == input.shape.dims, "Dynamic output shape is wrong");
    require(found->second.bytes == input.bytes, "Identity output values are wrong");

    TensorMap missing;
    auto rejected = backend.value()->infer(missing);
    require(!rejected && rejected.status().code == ErrorCode::InvalidArgument,
            "Missing ONNX input was not rejected");

    Tensor wrong = makeInput();
    wrong.shape.dims[1] = 1;
    wrong.bytes.resize(8 * sizeof(float));
    TensorMap wrongInputs;
    wrongInputs.emplace("input", std::move(wrong));
    rejected = backend.value()->infer(wrongInputs);
    require(!rejected && rejected.status().code == ErrorCode::TensorShapeMismatch,
            "Static ONNX dimension mismatch was not rejected");

}

void testInferenceSessionIntegration() {
    const auto package =
        std::filesystem::path(ANOM_TEST_DATA_DIR) / "direct_ort_package";
    LoadOptions options;
    options.warmup = true;
    auto session = InferenceSession::load(package, options);
    require(session.ok(), session.status().describe());
    require(session.value()->modelInfo().runtimeBackend == RuntimeBackend::OnnxRuntime,
            "InferenceSession did not report the ORT backend");
    require(session.value()->modelInfo().executionProvider == "cpu",
            "InferenceSession did not report the CPU execution provider");

    cv::Mat image(2, 2, CV_8UC3, cv::Scalar(10, 20, 30));
    auto prediction = session.value()->predict(image);
    require(prediction.ok(), prediction.status().describe());
    require(std::abs(prediction.value().rawScore - 0.4F) < 1e-6F,
            "Direct ORT score was not propagated through the adapter");
    require(prediction.value().rawAnomalyMap.size() == cv::Size(2, 2),
            "Direct ORT anomaly map has the wrong size");
    require(std::abs(prediction.value().rawAnomalyMap.at<float>(1, 1) - 0.4F) < 1e-6F,
            "Direct ORT anomaly map values are wrong");
}

}  // namespace

int main() {
    try {
        testIdentityModel();
        testInferenceSessionIntegration();
        std::cout << "ONNX Runtime backend tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ONNX Runtime backend tests failed: " << error.what() << '\n';
        return 1;
    }
}
