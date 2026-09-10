#include "retrieval/faiss_index.h"
#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>
#include <chrono>
#include <cmath>
#include <future>
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace anom::model;
namespace {
void require(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
struct Fixtures {
    std::filesystem::path root = std::filesystem::temp_directory_path() /
        ("anom-faiss-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    Fixtures() { std::filesystem::create_directory(root); }
    ~Fixtures() { std::error_code ignored; std::filesystem::remove_all(root, ignored); }
    std::unique_ptr<FaissIndex> index(const char* name, int dimension = 2) {
        auto cpu = std::make_unique<faiss::IndexFlatL2>(dimension);
        std::vector<float> values(static_cast<size_t>(4 * dimension));
        for (int i = 0; i < 4; ++i) for (int j = 0; j < dimension; ++j) values[i * dimension + j] = static_cast<float>(i * 3 + j);
        cpu->add(4, values.data());
        auto path = root / name;
        faiss::write_index(cpu.get(), path.string().c_str());
        return std::make_unique<FaissIndex>(std::move(cpu), path);
    }
};
int run() {
    Fixtures fixtures;
    auto index = fixtures.index("flat.faiss");
    auto second = fixtures.index("second.faiss");
    SearchExecutionConfig config;
    config.pluginDirectory = fixtures.root / "missing-plugins";
    config.deviceId = 123456;
    auto cpu = configureFaissIndexes({index.get(), second.get()}, config, 2);
    require(cpu.ok() && cpu.value().provider == ExecutionProvider::Cpu && !cpu.value().fallbackOccurred,
            "CPU mode touched GPU plugin/device");
    float queries[] = {0.2F, 1.0F, 8.9F, 10.1F};
    float expected[4];
    int64_t expectedLabels[4];
    index->search(2, queries, 2, expected, expectedLabels);
    require(expectedLabels[0] == 0 && expectedLabels[2] == 3, "CPU search returned incorrect neighbors");
    config.provider = ExecutionProvider::Cuda;
    auto strict = configureFaissIndexes({index.get()}, config, 2);
    require(!strict && strict.status().code == ErrorCode::PluginNotFound, "Strict GPU accepted a missing plugin");
    config.allowCpuFallback = true;
    auto fallback = configureFaissIndexes({index.get()}, config, 2);
    require(fallback.ok() && fallback.value().provider == ExecutionProvider::Cpu && fallback.value().fallbackOccurred &&
            !fallback.value().fallbackReason.empty(), "Missing GPU plugin did not report CPU fallback");
    config.pluginDirectory = ANOM_FAISS_TEST_PLUGIN_DIR;
    fallback = configureFaissIndexes({index.get()}, config, 2);
    require(fallback.ok() && fallback.value().fallbackOccurred, "Unavailable GPU device did not fall back");
    config.allowCpuFallback = false;
    require(!configureFaissIndexes({index.get()}, config, 2), "Strict GPU accepted unavailable device");
    config.deviceId = ANOM_FAISS_TEST_FAKE ? 1 : 0;
#if !ANOM_FAISS_TEST_FAKE
    auto library = plugins::DynamicLibrary::open(plugins::pluginPath(config.pluginDirectory, "search", "faiss_cuda"));
    require(library.ok(), "Configured GPU Faiss plugin could not be loaded");
    auto queryApi = reinterpret_cast<anom_faiss_query_v1_fn>(library.value()->symbol("anom_faiss_query_v1"));
    anom_faiss_api_v1 api{};
    api.struct_size = sizeof(api);
    require(queryApi && queryApi(1, &api) == 0, "Invalid GPU Faiss API");
    if (api.probe(0) != 0) { std::cout << "SKIP: CUDA device unavailable\n"; return 77; }
#endif
    auto gpu = configureFaissIndexes({index.get(), second.get()}, config, 2);
    if (!gpu) throw std::runtime_error(gpu.status().describe());
    require(gpu.value().provider == ExecutionProvider::Cuda && gpu.value().deviceId == config.deviceId,
            "Faiss GPU selection was not reported");
    std::async(std::launch::async, [&] {
        float actual[4];
        int64_t labels[4];
        index->search(2, queries, 2, actual, labels);
        for (int i = 0; i < 4; ++i) require(labels[i] == expectedLabels[i] && std::abs(actual[i] - expected[i]) < 1e-3F,
                                           "CPU/GPU distances or neighbor IDs differ");
        float support[2];
        index->reconstruct(labels[0], support);
        index->search(1, support, 2, actual, labels);
        require(labels[0] == 0 && std::abs(actual[0]) < 1e-3F, "PatchCore weighted support search failed");
    }).get();
#if ANOM_FAISS_TEST_FAKE
    // Runtime failures must be returned, even if initialization fallback was allowed.
    config.allowCpuFallback = true;
    require(configureFaissIndexes({index.get()}, config, 2).ok(), "GPU reload failed");
    float failing[] = {-999.0F, 0.0F};
    bool threw = false;
    try { index->search(1, failing, 2, expected, expectedLabels); } catch (const std::exception&) { threw = true; }
    require(threw, "Runtime search silently fell back to CPU");
    auto unsupported = fixtures.index("unsupported.faiss", 7);
    fallback = configureFaissIndexes({index.get(), unsupported.get()}, config, 2);
    require(fallback.ok() && fallback.value().provider == ExecutionProvider::Cpu && fallback.value().fallbackOccurred,
            "Multi-layer GPU initialization did not roll back to CPU");
    index->search(1, failing, 2, expected, expectedLabels); // CPU succeeds; proves first layer rolled back.
    config.allowCpuFallback = false;
    require(!configureFaissIndexes({index.get(), unsupported.get()}, config, 2), "Strict GPU accepted unsupported layer");
    auto corrupt = fixtures.index("corrupt.faiss");
    { std::ofstream stream(fixtures.root / "corrupt.faiss", std::ios::binary | std::ios::trunc); stream << "bad"; }
    config.allowCpuFallback = true;
    auto invalid = configureFaissIndexes({corrupt.get()}, config, 2);
    require(!invalid && invalid.status().code == ErrorCode::ArtifactCorrupt, "Corrupt artifact was hidden by CPU fallback");
#endif
    std::async(std::launch::async, [first = std::move(index), other = std::move(second)]() mutable {
        first.reset(); other.reset();
    }).get();
    std::cout << (ANOM_FAISS_TEST_FAKE ? "Faiss routing/lifetime tests (simulated GPU) passed\n" : "Real Faiss CPU/GPU parity passed\n");
    return 0;
}
}
int main() { try { return run(); } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; } }
