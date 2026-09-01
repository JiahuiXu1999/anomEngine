#include "adapters/adapter.h"
#include "adapters/feature_utils.h"
#include "infrastructure/sha256.h"
#include "infrastructure/utf8_path.h"
#include "model/manifest.h"
#include "pipeline/dfkde.h"
#include "pipeline/efficientad.h"
#include "pipeline/padim.h"
#include "pipeline/patchcore.h"
#include "pipeline/spade.h"
#include "processing/analyzer.h"
#include "processing/postprocessor.h"
#include "processing/preprocessor.h"

#include <faiss/IndexFlat.h>
#include <faiss/index_io.h>

#include <opencv2/core.hpp>

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace anom::model;

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

void testUtf8PathRoundTrip() {
    const std::u8string source = u8"模型/échantillon/данные.onnx";
    std::string encoded(source.size(), '\0');
    std::memcpy(encoded.data(), source.data(), source.size());
    require(pathToUtf8(pathFromUtf8(encoded)) == encoded,
            "UTF-8 filesystem path conversion did not preserve its code units");
}

void writeText(const fs::path& path, const std::string& content) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(stream), "Unable to create " + path.string());
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    require(static_cast<bool>(stream), "Unable to write " + path.string());
}

template <typename T>
void writeScalar(std::ofstream& stream, const T& value) {
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

struct TemporaryDirectory {
    fs::path path;
    TemporaryDirectory() {
        path = fs::temp_directory_path() / "anom_engine_model_contract_tests";
        std::error_code ignored;
        fs::remove_all(path, ignored);
        fs::create_directories(path);
    }
    ~TemporaryDirectory() {
        std::error_code ignored;
        fs::remove_all(path, ignored);
    }
};

Tensor scalarFeature(float value) {
    Tensor tensor;
    tensor.dtype = DataType::Float32;
    tensor.shape.dims = {1, 1, 1, 1};
    tensor.bytes.resize(sizeof(float));
    std::memcpy(tensor.bytes.data(), &value, sizeof(float));
    return tensor;
}

Tensor batchedFeature(const std::vector<float>& values) {
    Tensor tensor;
    tensor.dtype = DataType::Float32;
    tensor.shape.dims = {static_cast<std::int64_t>(values.size()), 1, 1, 1};
    tensor.bytes.resize(values.size() * sizeof(float));
    std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
    return tensor;
}

Tensor nchwFeature(const std::vector<float>& values, std::int64_t batch,
                   std::int64_t channels, std::int64_t height, std::int64_t width) {
    Tensor tensor;
    tensor.dtype = DataType::Float32;
    tensor.shape.dims = {batch, channels, height, width};
    tensor.bytes.resize(values.size() * sizeof(float));
    std::memcpy(tensor.bytes.data(), values.data(), tensor.bytes.size());
    return tensor;
}

void testExampleManifests() {
    const fs::path examples = fs::path(ANOM_SOURCE_DIR) / "src/model/examples";
    auto patchcore = ModelPackage::load(examples / "patchcore.manifest.json");
    require(patchcore.ok(), patchcore.status().describe());
    require(patchcore.value().manifest().algorithm == AlgorithmType::PatchCore,
            "PatchCore example selected wrong algorithm");
    auto padim = ModelPackage::load(examples / "padim.manifest.json");
    require(padim.ok(), padim.status().describe());
    require(padim.value().manifest().algorithm == AlgorithmType::Padim,
            "PaDiM example selected wrong algorithm");
    auto padimOrt = ModelPackage::load(examples / "padim-ort.manifest.json");
    require(padimOrt.ok(), padimOrt.status().describe());
    require(padimOrt.value().manifest().runtime.backend == RuntimeBackend::OnnxRuntime,
            "PaDiM ORT example selected wrong runtime backend");
    require(padimOrt.value().manifest().runtime.ortProvider == OrtExecutionProvider::Cpu,
            "PaDiM ORT example selected wrong execution provider");
    auto efficientad = ModelPackage::load(examples / "efficientad.manifest.json");
    require(efficientad.ok(), efficientad.status().describe());
    require(efficientad.value().manifest().algorithm == AlgorithmType::EfficientAD,
            "EfficientAD example selected wrong algorithm");
    require(efficientad.value().manifest().graphContract == GraphContract::Prediction,
            "EfficientAD example selected wrong graph contract");
    auto dfkde = ModelPackage::load(examples / "dfkde.manifest.json");
    require(dfkde.ok(), dfkde.status().describe());
    require(dfkde.value().manifest().algorithm == AlgorithmType::DFKDE,
            "DFKDE example selected wrong algorithm");
    auto spade = ModelPackage::load(examples / "spade.manifest.json");
    require(spade.ok(), spade.status().describe());
    require(spade.value().manifest().algorithm == AlgorithmType::SPADE,
            "SPADE example selected wrong algorithm");
    auto yolo = ModelPackage::load(examples / "yolo.manifest.json");
    require(yolo.ok(), yolo.status().describe());
    require(yolo.value().manifest().algorithm == AlgorithmType::Yolo,
            "YOLO example selected wrong algorithm");
    auto escaped = padim.value().resolveArtifact("../outside.bin", false);
    require(!escaped && escaped.status().code == ErrorCode::ArtifactPathEscape,
            "Artifact path traversal was not rejected");
}

void testPreprocessorAndPooling() {
    InputConfig config;
    config.resizeHeight = 4;
    config.resizeWidth = 4;
    config.centerCrop = cv::Size(2, 2);
    config.mean = {0.0F, 0.0F, 0.0F};
    config.std = {1.0F, 1.0F, 1.0F};
    ImagePreprocessor preprocessor(config);
    cv::Mat image(8, 6, CV_8UC3, cv::Scalar(0, 128, 255));
    auto batch = preprocessor.process({image});
    require(batch.ok(), batch.status().describe());
    require(batch.value().tensor.shape.dims == std::vector<std::int64_t>({1, 3, 2, 2}),
            "Preprocessor emitted unexpected tensor shape");
    require(batch.value().geometry.front().cropRect == cv::Rect(1, 1, 2, 2),
            "Preprocessor geometry did not record center crop");

    Tensor input;
    input.dtype = DataType::Float32;
    input.shape.dims = {1, 1, 2, 2};
    input.bytes.resize(4 * sizeof(float));
    const std::array<float, 4> values{1, 2, 3, 4};
    std::memcpy(input.bytes.data(), values.data(), input.bytes.size());
    auto pooled = feature::averagePool2d(input, 2, 2, 0);
    require(pooled.ok(), pooled.status().describe());
    require(std::abs(pooled.value().data<float>()[0] - 2.5F) < 1e-6F,
            "Average pooling result is incorrect");
}

void testPatchCoreAdapter(const fs::path& root) {
    faiss::IndexFlatL2 index(1);
    const float bank = 0.0F;
    index.add(1, &bank);
    faiss::write_index(&index, (root / "memory.faiss").string().c_str());
    writeText(root / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"test-patchcore","version":"1","algorithm":"patchcore"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"layer":"feature"},
      "algorithm":{"type":"patchcore","index":"memory.faiss","feature_layers":["layer"],
        "embedding_dimension":1,"num_neighbors":1,"pooling_kernel":1,"pooling_stride":1,
        "pooling_padding":0,"gaussian_sigma":0,"sqrt_distances":true},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(root);
    require(package.ok(), package.status().describe());
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());

    TensorSignature validSignature;
    validSignature.outputs.push_back({"feature", DataType::Float32, {{-1, 1, 1, 1}}, false});
    auto signatureStatus = adapter.value()->validateSignature(validSignature);
    require(signatureStatus.ok(), signatureStatus.status().describe());

    TensorSignature invalidSignature;
    invalidSignature.outputs.push_back({"feature", DataType::Float32, {{-1, 2, 1, 1}}, false});
    auto invalidStatus = adapter.value()->validateSignature(invalidSignature);
    require(!invalidStatus && invalidStatus.status().code == ErrorCode::TensorSignatureMismatch,
            "PatchCore accepted feature channels that disagree with the memory bank");

    TensorMap outputs;
    outputs.emplace("feature", batchedFeature({3.0F, 4.0F}));
    auto prediction = adapter.value()->predict(outputs, cv::Size(2, 2));
    require(prediction.ok(), prediction.status().describe());
    require(prediction.value().size() == 2, "PatchCore did not preserve the backend batch");
    require(std::abs(prediction.value().front().score - 3.0F) < 1e-6F,
            "PatchCore L2 distance was not converted to Euclidean distance");
    require(std::abs(prediction.value().back().score - 4.0F) < 1e-6F,
            "PatchCore returned the wrong score for the second image");
    require(prediction.value().front().anomalyMap.size() == cv::Size(2, 2),
            "PatchCore anomaly map has wrong size");
}

void testPadimAdapter(const fs::path& root) {
    const std::int32_t selectedChannel = 0;
    {
        std::ofstream indices(root / "channels.i32", std::ios::binary | std::ios::trunc);
        writeScalar(indices, selectedChannel);
    }
    {
        std::ofstream statistics(root / "stats.bin", std::ios::binary | std::ios::trunc);
        const std::array<char, 8> magic{'A', 'N', 'P', 'A', 'D', 'I', 'M', '\0'};
        statistics.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        const std::uint32_t one = 1;
        writeScalar(statistics, one);  // version
        writeScalar(statistics, one);  // height
        writeScalar(statistics, one);  // width
        writeScalar(statistics, one);  // dimension
        const float mean = 0.0F;
        const float precision = 1.0F;
        writeScalar(statistics, mean);
        writeScalar(statistics, precision);
    }
    writeText(root / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"test-padim","version":"1","algorithm":"padim"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"layer":"feature"},
      "algorithm":{"type":"padim","statistics":"stats.bin","channel_indices":"channels.i32",
        "feature_layers":["layer"],"embedding_dimension":1,"feature_map_size":[1,1],
        "gaussian_sigma":0},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(root);
    require(package.ok(), package.status().describe());
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());

    TensorSignature validSignature;
    validSignature.outputs.push_back({"feature", DataType::Float32, {{-1, 1, 1, 1}}, false});
    auto signatureStatus = adapter.value()->validateSignature(validSignature);
    require(signatureStatus.ok(), signatureStatus.status().describe());

    TensorSignature wrongSpatialSignature;
    wrongSpatialSignature.outputs.push_back(
        {"feature", DataType::Float32, {{-1, 1, 2, 1}}, false});
    auto wrongSpatialStatus = adapter.value()->validateSignature(wrongSpatialSignature);
    require(!wrongSpatialStatus &&
                wrongSpatialStatus.status().code == ErrorCode::TensorSignatureMismatch,
            "PaDiM accepted a feature map that disagrees with its statistics");

    TensorMap outputs;
    outputs.emplace("feature", batchedFeature({3.0F, 4.0F}));
    auto prediction = adapter.value()->predict(outputs, cv::Size(2, 2));
    require(prediction.ok(), prediction.status().describe());
    require(prediction.value().size() == 2, "PaDiM did not preserve the backend batch");
    require(std::abs(prediction.value().front().score - 3.0F) < 1e-6F,
            "PaDiM Mahalanobis score is incorrect");
    require(std::abs(prediction.value().back().score - 4.0F) < 1e-6F,
            "PaDiM returned the wrong score for the second image");

    const fs::path corruptRoot = root / "asymmetric";
    fs::create_directories(corruptRoot);
    {
        std::ofstream indices(corruptRoot / "channels.i32", std::ios::binary | std::ios::trunc);
        const std::array<std::int32_t, 2> selectedChannels{0, 1};
        indices.write(reinterpret_cast<const char*>(selectedChannels.data()),
                      static_cast<std::streamsize>(sizeof(selectedChannels)));
    }
    {
        std::ofstream statistics(corruptRoot / "stats.bin", std::ios::binary | std::ios::trunc);
        const std::array<char, 8> magic{'A', 'N', 'P', 'A', 'D', 'I', 'M', '\0'};
        statistics.write(magic.data(), static_cast<std::streamsize>(magic.size()));
        const std::uint32_t version = 1, height = 1, width = 1, dimension = 2;
        writeScalar(statistics, version);
        writeScalar(statistics, height);
        writeScalar(statistics, width);
        writeScalar(statistics, dimension);
        const std::array<float, 2> mean{0.0F, 0.0F};
        const std::array<float, 4> asymmetricPrecision{1.0F, 1.0F, 0.0F, 1.0F};
        statistics.write(reinterpret_cast<const char*>(mean.data()),
                         static_cast<std::streamsize>(sizeof(mean)));
        statistics.write(reinterpret_cast<const char*>(asymmetricPrecision.data()),
                         static_cast<std::streamsize>(sizeof(asymmetricPrecision)));
    }
    writeText(corruptRoot / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"corrupt-padim","version":"1","algorithm":"padim"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"layer":"feature"},
      "algorithm":{"type":"padim","statistics":"stats.bin","channel_indices":"channels.i32",
        "feature_layers":["layer"],"embedding_dimension":2,"feature_map_size":[1,1]},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto corruptPackage = ModelPackage::load(corruptRoot);
    require(corruptPackage.ok(), corruptPackage.status().describe());
    auto corruptAdapter = createAdapter(corruptPackage.value());
    require(!corruptAdapter && corruptAdapter.status().code == ErrorCode::ArtifactCorrupt,
            "PaDiM accepted an asymmetric precision matrix");
}

void testPostprocessor() {
    PostprocessConfig config;
    config.imageThreshold = 2.0F;
    config.imageMin = 0.0F;
    config.imageMax = 4.0F;
    config.pixelThreshold = 2.0F;
    config.pixelMin = 0.0F;
    config.pixelMax = 4.0F;
    AnomalyPostprocessor processor(config, "model", "1");
    RawPrediction raw;
    raw.score = 2.0F;
    raw.anomalyMap = cv::Mat(2, 2, CV_32FC1, cv::Scalar(2.0F));
    ImageGeometry geometry{cv::Size(4, 4), cv::Size(4, 4), cv::Rect(0, 0, 4, 4)};
    auto output = processor.process({raw}, {geometry});
    require(output.ok(), output.status().describe());
    require(std::abs(output.value().front().score - 0.5F) < 1e-6F,
            "Threshold-centered normalization is incorrect");
    require(output.value().front().isAnomalous, "Score at threshold should be anomalous");
    require(output.value().front().mask.size() == cv::Size(4, 4), "Mask geometry restoration failed");
}

void testAnalyzer() {
    cv::Mat map(10, 10, CV_32FC1, cv::Scalar(0.5F));
    cv::Mat mask = cv::Mat::zeros(10, 10, CV_8UC1);
    for (int y = 1; y <= 2; ++y) {
        for (int x = 1; x <= 2; ++x) {
            map.at<float>(y, x) = 0.9F;
            mask.at<uchar>(y, x) = 255;
        }
    }
    map.at<float>(8, 8) = 0.9F;
    mask.at<uchar>(8, 8) = 255;

    AnomalyAnalyzer analyzer(AnalysisConfig{1, 0});
    auto result = analyzer.analyze(map, mask);
    require(result.ok(), result.status().describe());
    const AnomalyAnalysis& analysis = result.value();
    require(analysis.hasMap, "Analysis did not record the anomaly map");
    require(analysis.regionCount == 2, "Analyzer found the wrong number of regions");
    require(std::abs(analysis.anomalyAreaRatio - 0.05F) < 1e-6F, "Anomaly area ratio is incorrect");
    require(std::abs(analysis.maxScore - 0.9F) < 1e-6F, "Peak score is incorrect");
    require(analysis.regions.front().area == 4.0, "Regions are not sorted by descending area");
    require(analysis.regions.front().boundingBox == cv::Rect(1, 1, 2, 2), "Region bounding box is wrong");
}

void testAnalyzeThroughPostprocessor() {
    PostprocessConfig config;
    config.threshold = true;
    config.analyze = true;
    config.pixelThreshold = 0.5F;
    config.pixelMin = 0.0F;
    config.pixelMax = 1.0F;
    config.minRegionArea = 1;
    AnomalyPostprocessor processor(config, "model", "1");

    cv::Mat map(4, 4, CV_32FC1, cv::Scalar(0.0F));
    map.at<float>(0, 0) = 1.0F;
    map.at<float>(2, 2) = 1.0F;

    RawPrediction raw;
    raw.score = 1.0F;
    raw.anomalyMap = map;
    ImageGeometry geometry{cv::Size(4, 4), cv::Size(4, 4), cv::Rect(0, 0, 4, 4)};
    auto output = processor.process({raw}, {geometry});
    require(output.ok(), output.status().describe());
    const Prediction& prediction = output.value().front();
    require(prediction.analysis.hasMap, "Postprocessor did not populate analysis");
    require(prediction.analysis.regionCount == 2, "Postprocessor analysis found the wrong region count");
    require(std::abs(prediction.analysis.anomalyAreaRatio - 0.125F) < 1e-6F,
            "Postprocessor analysis area ratio is incorrect");
}

void testTrainingPipelines(const fs::path& root) {
    const fs::path patchcoreRoot = root / "patchcore-training";
    fs::create_directories(patchcoreRoot);
    PatchCorePipeline patchcore({1, 0.5F, 1, 1'000, 42});
    auto patchAdded = patchcore.addEmbedding(batchedFeature({0.0F, 1.0F, 5.0F, 6.0F}));
    require(patchAdded.ok(), patchAdded.status().describe());
    require(patchcore.featureCount() == 4, "PatchCore pipeline lost training features");
    auto savedBank = patchcore.saveMemoryBank(patchcoreRoot / "memory.faiss");
    require(savedBank.ok(), savedBank.status().describe());
    require(savedBank.value() == 2, "PatchCore pipeline selected the wrong coreset size");
    std::unique_ptr<faiss::Index> index(
        faiss::read_index((patchcoreRoot / "memory.faiss").string().c_str()));
    require(index && index->d == 1 && index->ntotal == 2,
            "PatchCore pipeline wrote an invalid FAISS index");
    require(patchcore.saveCheckpoint(patchcoreRoot / "checkpoint.bin").ok(),
            "PatchCore pipeline could not save a checkpoint");
    PatchCorePipeline resumedPatchcore({1, 0.5F, 1, 1'000, 42});
    require(resumedPatchcore.loadCheckpoint(patchcoreRoot / "checkpoint.bin").ok(),
            "PatchCore pipeline could not load a checkpoint");
    require(resumedPatchcore.featureCount() == patchcore.featureCount(),
            "PatchCore checkpoint lost training features");
    PatchCorePipeline importedPatchcore({1, 1.0F, 1, 1'000, 42});
    auto imported = importedPatchcore.loadMemoryBank(patchcoreRoot / "memory.faiss");
    require(imported.ok() && imported.value() == 2,
            "PatchCore pipeline could not import an existing memory bank");

    const fs::path padimRoot = root / "padim-training";
    fs::create_directories(padimRoot);
    PadimPipeline padim({1, 1, 1, 1.0, {0}});
    auto padimAdded = padim.addEmbedding(batchedFeature({1.0F, 3.0F}));
    require(padimAdded.ok(), padimAdded.status().describe());
    require(padim.sampleCount() == 2, "PaDiM pipeline lost training samples");
    auto savedStatistics = padim.saveArtifacts(
        padimRoot / "stats.bin", padimRoot / "channels.i32");
    require(savedStatistics.ok(), savedStatistics.status().describe());
    require(padim.saveCheckpoint(padimRoot / "checkpoint.bin").ok(),
            "PaDiM pipeline could not save a checkpoint");
    PadimPipeline resumedPadim({1, 1, 1, 1.0, {0}});
    require(resumedPadim.loadCheckpoint(padimRoot / "checkpoint.bin").ok(),
            "PaDiM pipeline could not load a checkpoint");
    require(resumedPadim.sampleCount() == padim.sampleCount(),
            "PaDiM checkpoint lost its sample count");
    writeText(padimRoot / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"trained-padim","version":"1","algorithm":"padim"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"layer":"feature"},
      "algorithm":{"type":"padim","statistics":"stats.bin","channel_indices":"channels.i32",
        "feature_layers":["layer"],"embedding_dimension":1,"feature_map_size":[1,1],
        "gaussian_sigma":0},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(padimRoot);
    require(package.ok(), package.status().describe());
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());
    TensorMap output;
    output.emplace("feature", scalarFeature(4.0F));
    auto prediction = adapter.value()->predict(output, cv::Size(1, 1));
    require(prediction.ok(), prediction.status().describe());
    require(std::abs(prediction.value().front().score - std::sqrt(2.0F)) < 1.0e-5F,
            "PaDiM pipeline artifacts are incompatible with PadimAdapter");
}

void testEfficientAD(const fs::path& root) {
    const fs::path adapterRoot = root / "efficientad";
    fs::create_directories(adapterRoot);
    writeText(adapterRoot / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"test-efficientad","version":"1","algorithm":"efficientad"},
      "graph_contract":"prediction",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"teacher":"t","student":"s","reconstruction_error":"r"},
      "algorithm":{"type":"efficientad","teacher_semantic":"teacher",
        "student_semantic":"student","reconstruction_error_semantic":"reconstruction_error",
        "hard_weight":1.0,"soft_weight":1.0,"gaussian_sigma":0},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(adapterRoot);
    require(package.ok(), package.status().describe());
    require(package.value().manifest().algorithm == AlgorithmType::EfficientAD,
            "EfficientAD manifest selected the wrong algorithm");
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());

    TensorSignature signature;
    signature.outputs.push_back({"t", DataType::Float32, {{-1, 1, 2, 2}}, false});
    signature.outputs.push_back({"s", DataType::Float32, {{-1, 1, 2, 2}}, false});
    signature.outputs.push_back({"r", DataType::Float32, {{-1, 1, 2, 2}}, false});
    auto signatureStatus = adapter.value()->validateSignature(signature);
    require(signatureStatus.ok(), signatureStatus.status().describe());

    TensorSignature mismatchedSignature;
    mismatchedSignature.outputs.push_back({"t", DataType::Float32, {{-1, 1, 2, 2}}, false});
    mismatchedSignature.outputs.push_back({"s", DataType::Float32, {{-1, 2, 2, 2}}, false});
    mismatchedSignature.outputs.push_back({"r", DataType::Float32, {{-1, 1, 2, 2}}, false});
    auto mismatchStatus = adapter.value()->validateSignature(mismatchedSignature);
    require(!mismatchStatus &&
                mismatchStatus.status().code == ErrorCode::TensorSignatureMismatch,
            "EfficientAD accepted teacher/student streams with mismatched channels");

    TensorMap outputs;
    outputs.emplace("t", nchwFeature({1.0F, 2.0F, 3.0F, 4.0F}, 1, 1, 2, 2));
    outputs.emplace("s", nchwFeature({0.0F, 0.0F, 0.0F, 0.0F}, 1, 1, 2, 2));
    outputs.emplace("r", nchwFeature({0.0F, 0.0F, 0.0F, 0.0F}, 1, 1, 2, 2));
    auto prediction = adapter.value()->predict(outputs, cv::Size(2, 2));
    require(prediction.ok(), prediction.status().describe());
    require(prediction.value().size() == 1, "EfficientAD did not preserve the backend batch");
    require(std::abs(prediction.value().front().score - 0.5F) < 1.0e-5F,
            "EfficientAD combined score is incorrect");
    require(prediction.value().front().anomalyMap.size() == cv::Size(2, 2),
            "EfficientAD anomaly map has wrong size");

    EfficientADPipeline pipeline({1.0F, 1.0F, 0.0F});
    EfficientADConfig algorithm;
    std::unordered_map<std::string, std::string> bindings{
        {"teacher", "t"}, {"student", "s"}, {"reconstruction_error", "r"}};
    auto added = pipeline.addBackendOutputs(outputs, bindings, algorithm, cv::Size(2, 2));
    require(added.ok(), added.status().describe());
    require(pipeline.sampleCount() == 1, "EfficientAD pipeline lost training samples");
    auto saved = pipeline.saveStatistics(adapterRoot / "stats.bin");
    require(saved.ok(), saved.status().describe());
    require(std::filesystem::is_regular_file(adapterRoot / "stats.bin"),
            "EfficientAD pipeline did not write a statistics artifact");
}

void testDFKDE(const fs::path& root) {
    const fs::path dfkdeRoot = root / "dfkde";
    fs::create_directories(dfkdeRoot);

    DfkdePipeline pipeline({1, 1, 1.0F});
    auto added = pipeline.addEmbedding(batchedFeature({1.0F, 3.0F}));
    require(added.ok(), added.status().describe());
    require(pipeline.featureCount() == 2, "DFKDE pipeline lost training features");
    auto saved = pipeline.saveStatistics(dfkdeRoot / "stats.bin");
    require(saved.ok(), saved.status().describe());

    writeText(dfkdeRoot / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"test-dfkde","version":"1","algorithm":"dfkde"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"layer":"f"},
      "algorithm":{"type":"dfkde","statistics":"stats.bin","feature_layers":["layer"],
        "embedding_dimension":1,"n_components":1,"kernel_sigma":1.0,"gaussian_sigma":0},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(dfkdeRoot);
    require(package.ok(), package.status().describe());
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());

    TensorSignature signature;
    signature.outputs.push_back({"f", DataType::Float32, {{-1, 1, 1, 1}}, false});
    require(adapter.value()->validateSignature(signature).ok(), "DFKDE rejected a valid signature");

    TensorMap outputs;
    outputs.emplace("f", batchedFeature({1.0F, 10.0F}));
    auto prediction = adapter.value()->predict(outputs, cv::Size(2, 2));
    require(prediction.ok(), prediction.status().describe());
    require(prediction.value().size() == 2, "DFKDE did not preserve the backend batch");
    require(prediction.value().front().score < prediction.value().back().score,
            "DFKDE should score the distant patch higher");
    require(prediction.value().front().anomalyMap.size() == cv::Size(2, 2),
            "DFKDE anomaly map has wrong size");
}

void testYolo(const fs::path& root) {
    const fs::path yoloRoot = root / "yolo";
    fs::create_directories(yoloRoot);
    writeText(yoloRoot / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"test-yolo","version":"1","algorithm":"yolo"},
      "graph_contract":"prediction",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[64,64]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"det":"output0"},
      "algorithm":{"type":"yolo","detection_semantic":"det","num_classes":1,
        "conf_threshold":0.25,"nms_threshold":0.45},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(yoloRoot);
    require(package.ok(), package.status().describe());
    require(package.value().manifest().algorithm == AlgorithmType::Yolo,
            "YOLO manifest selected the wrong algorithm");
    require(package.value().manifest().graphContract == GraphContract::Prediction,
            "YOLO manifest selected the wrong graph contract");
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());

    TensorSignature signature;
    signature.outputs.push_back({"output0", DataType::Float32, {{-1, 5, -1}}, false});
    require(adapter.value()->validateSignature(signature).ok(), "YOLO rejected a valid signature");

    TensorSignature wrongChannels;
    wrongChannels.outputs.push_back({"output0", DataType::Float32, {{-1, 6, -1}}, false});
    auto wrongStatus = adapter.value()->validateSignature(wrongChannels);
    require(!wrongStatus && wrongStatus.status().code == ErrorCode::TensorSignatureMismatch,
            "YOLO accepted channels that disagree with num_classes");

    // Detection tensor [1, 5, 3]: cx / cy / w / h / class-score per anchor.
    // box0 (0.9) and box1 (0.5) survive, box2 (0.1) is below the threshold.
    const std::vector<float> detectionData{
        10.0F, 20.0F, 30.0F,   // cx
        10.0F, 20.0F, 30.0F,   // cy
         4.0F,  4.0F,  4.0F,   // w
         4.0F,  4.0F,  4.0F,   // h
         0.9F,  0.5F,  0.1F};  // class score
    Tensor detections;
    detections.dtype = DataType::Float32;
    detections.shape.dims = {1, 5, 3};
    detections.bytes.resize(detectionData.size() * sizeof(float));
    std::memcpy(detections.bytes.data(), detectionData.data(), detections.bytes.size());
    TensorMap outputs;
    outputs.emplace("output0", detections);
    auto prediction = adapter.value()->predict(outputs, cv::Size(64, 64));
    require(prediction.ok(), prediction.status().describe());
    require(prediction.value().size() == 1, "YOLO did not preserve the backend batch");
    const auto& raw = prediction.value().front();
    require(std::abs(raw.score - 0.9F) < 1e-6F,
            "YOLO image score should be the highest confidence");
    require(raw.anomalyMap.size() == cv::Size(64, 64), "YOLO anomaly map has wrong size");
    require(std::abs(raw.anomalyMap.at<float>(10, 10) - 0.9F) < 1e-6F,
            "YOLO did not paint the high-confidence box");
    require(std::abs(raw.anomalyMap.at<float>(20, 20) - 0.5F) < 1e-6F,
            "YOLO did not paint the medium-confidence box");
    require(std::abs(raw.anomalyMap.at<float>(30, 30)) < 1e-6F,
            "YOLO painted a box below the confidence threshold");

    // NMS: two heavily overlapping boxes; the lower-scoring one is suppressed.
    const std::vector<float> overlapData{
        10.0F, 12.0F,   // cx
        10.0F, 12.0F,   // cy
        10.0F, 10.0F,   // w
        10.0F, 10.0F,   // h
         0.9F,  0.5F};  // class score
    Tensor overlap;
    overlap.dtype = DataType::Float32;
    overlap.shape.dims = {1, 5, 2};
    overlap.bytes.resize(overlapData.size() * sizeof(float));
    std::memcpy(overlap.bytes.data(), overlapData.data(), overlap.bytes.size());
    TensorMap overlapOutputs;
    overlapOutputs.emplace("output0", overlap);
    auto overlapPrediction = adapter.value()->predict(overlapOutputs, cv::Size(64, 64));
    require(overlapPrediction.ok(), overlapPrediction.status().describe());
    const auto& overlapRaw = overlapPrediction.value().front();
    require(std::abs(overlapRaw.score - 0.9F) < 1e-6F,
            "YOLO NMS should keep the highest-confidence box");
    require(std::abs(overlapRaw.anomalyMap.at<float>(16, 16)) < 1e-6F,
            "YOLO NMS failed to suppress the lower-confidence overlapping box");
}

void testSPADE(const fs::path& root) {
    const fs::path spadeRoot = root / "spade";
    fs::create_directories(spadeRoot);

    SpadePipeline pipeline;
    TensorMap trainOutputs;
    trainOutputs.emplace("f", scalarFeature(0.0F));
    std::unordered_map<std::string, std::string> bindings{{"layer", "f"}};
    SPADEConfig algorithm;
    algorithm.featureLayers = {"layer"};
    algorithm.indexFiles = {"spade.faiss"};
    auto added = pipeline.addBackendOutputs(trainOutputs, bindings, algorithm);
    require(added.ok(), added.status().describe());
    auto saved = pipeline.saveIndexes({spadeRoot / "spade.faiss"});
    require(saved.ok(), saved.status().describe());

    writeText(spadeRoot / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"test-spade","version":"1","algorithm":"spade"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"layer":"f"},
      "algorithm":{"type":"spade","feature_layers":["layer"],"indexes":["spade.faiss"],
        "num_neighbors":1,"gaussian_sigma":0},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON");
    auto package = ModelPackage::load(spadeRoot);
    require(package.ok(), package.status().describe());
    auto adapter = createAdapter(package.value());
    require(adapter.ok(), adapter.status().describe());

    TensorSignature signature;
    signature.outputs.push_back({"f", DataType::Float32, {{-1, 1, 1, 1}}, false});
    require(adapter.value()->validateSignature(signature).ok(), "SPADE rejected a valid signature");

    TensorMap outputs;
    outputs.emplace("f", batchedFeature({1.0F, 10.0F}));
    auto prediction = adapter.value()->predict(outputs, cv::Size(2, 2));
    require(prediction.ok(), prediction.status().describe());
    require(prediction.value().size() == 2, "SPADE did not preserve the backend batch");
    require(prediction.value().front().score < prediction.value().back().score,
            "SPADE should score the distant patch higher");
    require(prediction.value().front().anomalyMap.size() == cv::Size(2, 2),
            "SPADE anomaly map has wrong size");
}

void testSha256(const fs::path& root) {
    writeText(root / "abc.txt", "abc");
    auto digest = sha256File(root / "abc.txt");
    require(digest.ok(), digest.status().describe());
    require(digest.value() == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad",
            "SHA-256 implementation returned an incorrect digest");
    writeText(root / "manifest.json", R"JSON({
      "schema_version":1,
      "model":{"id":"checksum-test","version":"1","algorithm":"direct"},
      "graph_contract":"prediction",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2]},
      "runtime":{"onnx":"unused.onnx"},
      "outputs":{"pred_score":"score"},
      "algorithm":{"type":"direct"},
      "checksums":{"abc.txt":"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"}
    })JSON");
    auto validPackage = ModelPackage::load(root);
    require(validPackage.ok(), validPackage.status().describe());
    writeText(root / "abc.txt", "tampered");
    auto invalidPackage = ModelPackage::load(root);
    require(!invalidPackage && invalidPackage.status().code == ErrorCode::ArtifactCorrupt,
            "Tampered artifact was not rejected");
}

}  // namespace

int main() {
    try {
        testUtf8PathRoundTrip();
        testExampleManifests();
        testPreprocessorAndPooling();
        testPostprocessor();
        testAnalyzer();
        testAnalyzeThroughPostprocessor();
        TemporaryDirectory temporary;
        fs::create_directories(temporary.path / "patchcore");
        fs::create_directories(temporary.path / "padim");
        testSha256(temporary.path);
        testPatchCoreAdapter(temporary.path / "patchcore");
        testPadimAdapter(temporary.path / "padim");
        testTrainingPipelines(temporary.path);
        testEfficientAD(temporary.path);
        testDFKDE(temporary.path);
        testSPADE(temporary.path);
        testYolo(temporary.path);
        std::cout << "model contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "model contract tests failed: " << error.what() << '\n';
        return 1;
    }
}
