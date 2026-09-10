#include "backends/backend_factory.h"
#include "plugins/dynamic_library.h"
#include "plugin_api/anom_backend_plugin.h"

#include <future>
#include <iostream>
#include <stdexcept>
#include <utility>

using namespace anom::model;
namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void run() {
    const auto directory = std::filesystem::path(ANOM_MEMORY_PLUGIN_DIR);
    auto library = plugins::DynamicLibrary::open(plugins::pluginPath(directory, "backend", "onnxruntime"));
    require(library.ok(), "Unable to load test plugin");
    auto stat = reinterpret_cast<int (ANOM_PLUGIN_CALL *)(int)>(library.value()->symbol("memory_stat"));
    auto lastData = reinterpret_cast<const void* (ANOM_PLUGIN_CALL *)()>(library.value()->symbol("memory_last_data"));
    require(stat && lastData, "Missing test hooks");
    auto backend = createRuntimeBackend(RuntimeBackend::OnnxRuntime, directory);
    require(backend.ok(), "Unable to create backend");
    BackendConfig config;
    config.backend = RuntimeBackend::OnnxRuntime;
    config.provider = ExecutionProvider::Cpu;
    require(backend.value()->load(config).ok(), "Unable to load backend");

    auto first = backend.value()->infer({});
    require(first.ok(), "Inference failed");
    const Tensor& output = first.value().at("output");
    require(output.data<float>() == lastData(), "Feature buffer was copied across the plugin boundary");
    require(output.bytes.empty() && output.byteSize() == 4 * 1024 * 1024, "Incorrect shared feature buffer");
    Tensor retained = output;
    const auto* originalPointer = output.data<float>();
    auto next = backend.value()->infer({});
    require(next.ok(), "Second inference failed");
    require(std::as_const(retained).data<float>() == originalPointer && originalPointer[100] == 3.25F,
            "Later inference invalidated the first batch");

    Tensor writable = retained;
    writable.data<float>()[100] = 9.0F;
    require(originalPointer[100] == 3.25F && !writable.externalOwner,
            "Mutable access did not isolate shared output");
    first.value().clear();
    next.value().clear();
    require(stat(1) == 1 && stat(2) == 1, "Batch copy ownership or release count is wrong");

    TensorMap invalid;
    invalid.emplace("invalid", Tensor{});
    auto rejected = backend.value()->infer(invalid);
    require(!rejected && rejected.status().code == ErrorCode::BackendFailure,
            "Malformed output was accepted");
    require(stat(1) == 1 && stat(2) == 2, "Partial conversion leaked or released a batch twice");
    TensorMap empty;
    empty.emplace("empty", Tensor{});
    auto emptyResult = backend.value()->infer(empty);
    require(emptyResult.ok() && emptyResult.value().empty() && stat(1) == 1 && stat(2) == 3,
            "Empty batch ownership leaked");

    backend.value().reset();
    require(stat(0) == 1 && originalPointer[100] == 3.25F,
            "Backend instance was destroyed before the retained output");
    std::async(std::launch::async, [tensor = std::move(retained)]() mutable { tensor = {}; }).get();
    require(stat(0) == 0 && stat(1) == 0 && stat(2) == 4 && stat(3) == 0,
            "Cross-thread release order or exactly-once destruction is wrong");

    // Drop all explicit DLL handles too; the tensor must retain its release code.
    backend = createRuntimeBackend(RuntimeBackend::OnnxRuntime, directory);
    require(backend.ok() && backend.value()->load(config).ok(), "Reload failed");
    auto result = backend.value()->infer({});
    require(result.ok(), "Reload inference failed");
    retained = result.value().at("output");
    result.value().clear();
    backend.value().reset();
    library.value().reset();
    require(std::as_const(retained).data<float>()[100] == 3.25F, "DLL lifetime was not retained");
    std::async(std::launch::async, [tensor = std::move(retained)]() mutable { tensor = {}; }).get();
}
}
int main() {
    try {
        run();
        std::cout << "Shared backend memory: pointer identity, copy-on-write, repeated inference, malformed/empty batches, and cross-thread lifetime passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
