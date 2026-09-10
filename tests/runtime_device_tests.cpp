#include "backends/backend_factory.h"

#include <cstring>
#include <filesystem>
#include <future>
#include <iostream>
#include <stdexcept>

using namespace anom::model;

namespace {
void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

TensorMap inputs(int batch) {
    Tensor tensor;
    tensor.dtype = DataType::Float32;
    tensor.shape.dims = {batch, 3, 2, 2};
    tensor.bytes.resize(static_cast<std::size_t>(batch) * 12 * sizeof(float));
    for (int i = 0; i < batch * 12; ++i) tensor.data<float>()[i] = static_cast<float>(i) * 0.25F;
    TensorMap result;
    result.emplace("input", std::move(tensor));
    return result;
}
}

int main(int argc, char** argv) {
    try {
        require(argc == 2, "Expected tensorrt or onnxruntime");
        const RuntimeBackend kind = std::strcmp(argv[1], "tensorrt") == 0
            ? RuntimeBackend::TensorRT : RuntimeBackend::OnnxRuntime;
        auto gpu = createRuntimeBackend(kind);
        if (!gpu) {
            std::cout << "SKIP: " << gpu.status().describe() << '\n';
            return 77;
        }
        auto available = gpu.value()->probe(ExecutionProvider::Cuda, 0);
        if (!available) {
            std::cout << "SKIP: " << available.status().describe() << '\n';
            return 77;
        }
        BackendConfig config;
        config.backend = kind;
        config.provider = ExecutionProvider::Cuda;
        config.onnxPath = std::filesystem::path(ANOM_TEST_DATA_DIR) / "identity_dynamic.onnx";
        config.loadPolicy = EngineLoadPolicy::BuildIfMissing;
        config.inputHeight = 2;
        config.inputWidth = 2;
        config.maxBatchSize = 4;
        config.workspaceBytes = 64 * 1024 * 1024;
        auto loaded = gpu.value()->load(config);
        require(loaded.ok(), loaded.status().describe());

        auto cpu = createRuntimeBackend(RuntimeBackend::OnnxRuntime);
        require(cpu.ok(), cpu.status().describe());
        config.backend = RuntimeBackend::OnnxRuntime;
        config.provider = ExecutionProvider::Cpu;
        loaded = cpu.value()->load(config);
        require(loaded.ok(), loaded.status().describe());

        Tensor retainedGpuOutput;
        Tensor retainedCpuOutput;
        for (int batch : {2, 1, 4, 1}) {
            auto input = inputs(batch);
            auto expected = cpu.value()->infer(input);
            require(expected.ok(), expected.status().describe());
            // Exercise invocation on a different host thread than load().
            auto actual = std::async(std::launch::async, [&] { return gpu.value()->infer(input); }).get();
            require(actual.ok(), actual.status().describe());
            const auto& reference = expected.value().at("output");
            const auto& output = actual.value().at("output");
            require(reference.shape.dims == output.shape.dims &&
                        reference.byteSize() == output.byteSize() &&
                        std::memcmp(reference.data<std::byte>(), output.data<std::byte>(), output.byteSize()) == 0,
                    "CPU/GPU dynamic-batch identity results differ");
            retainedGpuOutput = output;
            retainedCpuOutput = reference;
        }
        // Retained outputs keep the instance/DLL alive beyond the wrapper.
        std::async(std::launch::async, [runtime = std::move(gpu.value())]() mutable { runtime.reset(); }).get();
        require(retainedGpuOutput.byteSize() == retainedCpuOutput.byteSize() &&
                    std::memcmp(std::as_const(retainedGpuOutput).data<std::byte>(),
                                std::as_const(retainedCpuOutput).data<std::byte>(),
                                retainedGpuOutput.byteSize()) == 0,
                "Retained GPU output changed after wrapper destruction");
        // Final batch and instance destruction must select the owning device.
        std::async(std::launch::async, [tensor = std::move(retainedGpuOutput)]() mutable { tensor = {}; }).get();
        std::cout << argv[1] << " CUDA parity and cross-thread lifetime checks passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
