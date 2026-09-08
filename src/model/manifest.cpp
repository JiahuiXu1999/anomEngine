#include "model/manifest.h"

#include "infrastructure/utf8_path.h"
#include "infrastructure/json.h"
#include "infrastructure/sha256.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace anom::model {
namespace {

class ManifestError final : public std::runtime_error {
public:
    ManifestError(std::string message, std::string path)
        : std::runtime_error(std::move(message)), path(std::move(path)) {}
    std::string path;
};

class Reader {
public:
    Reader(const json::Value& value, std::string path = "$") : value_(value), path_(std::move(path)) {}

    Reader required(const std::string& key) const {
        object();
        const auto* value = value_.find(key);
        if (!value) fail("Missing required property", childPath(key));
        return Reader(*value, childPath(key));
    }

    std::optional<Reader> optional(const std::string& key) const {
        object();
        const auto* value = value_.find(key);
        if (!value || value->isNull()) return std::nullopt;
        return Reader(*value, childPath(key));
    }

    const json::Value::Object& object() const {
        if (!value_.isObject()) fail("Expected object", path_);
        return value_.asObject();
    }

    const json::Value::Array& array() const {
        if (!value_.isArray()) fail("Expected array", path_);
        return value_.asArray();
    }

    std::string string() const {
        if (!value_.isString()) fail("Expected string", path_);
        return value_.asString();
    }

    bool boolean() const {
        if (!value_.isBool()) fail("Expected boolean", path_);
        return value_.asBool();
    }

    double number() const {
        if (!value_.isNumber()) fail("Expected number", path_);
        return value_.asNumber();
    }

    int integer() const {
        const double value = number();
        if (std::floor(value) != value || value < std::numeric_limits<int>::min() ||
            value > std::numeric_limits<int>::max()) {
            fail("Expected 32-bit integer", path_);
        }
        return static_cast<int>(value);
    }

    float real() const { return static_cast<float>(number()); }

    std::vector<std::string> stringArray() const {
        std::vector<std::string> output;
        const auto& values = array();
        output.reserve(values.size());
        for (std::size_t i = 0; i < values.size(); ++i) {
            output.push_back(Reader(values[i], path_ + "[" + std::to_string(i) + "]").string());
        }
        return output;
    }

    std::array<int, 2> intPair() const {
        const auto& values = array();
        if (values.size() != 2) fail("Expected an array of two integers", path_);
        return {Reader(values[0], path_ + "[0]").integer(),
                Reader(values[1], path_ + "[1]").integer()};
    }

    std::array<float, 3> floatTriple() const {
        const auto& values = array();
        if (values.size() != 3) fail("Expected an array of three numbers", path_);
        return {Reader(values[0], path_ + "[0]").real(),
                Reader(values[1], path_ + "[1]").real(),
                Reader(values[2], path_ + "[2]").real()};
    }

private:
    [[noreturn]] static void fail(const std::string& message, const std::string& path) {
        throw ManifestError(message, path);
    }

    std::string childPath(const std::string& key) const { return path_ + "." + key; }

    const json::Value& value_;
    std::string path_;
};

template <typename T>
T optionalOr(const Reader& reader, const std::string& key, T fallback,
             T (Reader::*getter)() const) {
    const auto child = reader.optional(key);
    return child ? ((*child).*getter)() : std::move(fallback);
}

std::filesystem::path optionalPath(const Reader& reader, const std::string& key) {
    const auto value = reader.optional(key);
    return value ? pathFromUtf8(value->string()) : std::filesystem::path{};
}

AlgorithmType parseAlgorithm(const std::string& value) {
    if (value == "patchcore") return AlgorithmType::PatchCore;
    if (value == "padim") return AlgorithmType::Padim;
    if (value == "direct") return AlgorithmType::Direct;
    if (value == "efficientad") return AlgorithmType::EfficientAD;
    if (value == "dfkde") return AlgorithmType::DFKDE;
    if (value == "spade") return AlgorithmType::SPADE;
    if (value == "yolo") return AlgorithmType::Yolo;
    throw ManifestError("Unsupported algorithm '" + value + "'", "$.model.algorithm");
}

GraphContract parseContract(const std::string& value) {
    if (value == "feature_pyramid") return GraphContract::FeaturePyramid;
    if (value == "prediction") return GraphContract::Prediction;
    throw ManifestError("Unsupported graph contract '" + value + "'", "$.graph_contract");
}

EngineLoadPolicy parseLoadPolicy(const std::string& value) {
    if (value == "engine_only") return EngineLoadPolicy::EngineOnly;
    if (value == "prefer_engine") return EngineLoadPolicy::PreferEngine;
    if (value == "build_if_missing") return EngineLoadPolicy::BuildIfMissing;
    throw ManifestError("Unsupported engine load policy '" + value + "'", "$.runtime.load_policy");
}

RuntimeBackend parseRuntimeBackend(const std::string& value) {
    if (value == "tensorrt") return RuntimeBackend::TensorRT;
    if (value == "onnxruntime" || value == "ort") return RuntimeBackend::OnnxRuntime;
    throw ManifestError("Unsupported runtime backend '" + value + "'", "$.runtime.backend");
}

OrtGraphOptimization parseOrtGraphOptimization(const std::string& value) {
    if (value == "disabled") return OrtGraphOptimization::Disabled;
    if (value == "basic") return OrtGraphOptimization::Basic;
    if (value == "extended") return OrtGraphOptimization::Extended;
    if (value == "all") return OrtGraphOptimization::All;
    throw ManifestError("Unsupported ONNX Runtime graph optimization level '" + value + "'",
                        "$.runtime.graph_optimization");
}

OrtExecutionMode parseOrtExecutionMode(const std::string& value) {
    if (value == "sequential") return OrtExecutionMode::Sequential;
    if (value == "parallel") return OrtExecutionMode::Parallel;
    throw ManifestError("Unsupported ONNX Runtime execution mode '" + value + "'",
                        "$.runtime.execution_mode");
}

TensorLayout parseLayout(const std::string& value) {
    if (value == "NCHW") return TensorLayout::NCHW;
    if (value == "NHWC") return TensorLayout::NHWC;
    throw ManifestError("Only NCHW and NHWC input layouts are supported", "$.input.layout");
}

int parseInterpolation(const std::string& value) {
    if (value == "nearest") return cv::INTER_NEAREST;
    if (value == "linear") return cv::INTER_LINEAR;
    if (value == "cubic") return cv::INTER_CUBIC;
    if (value == "area") return cv::INTER_AREA;
    throw ManifestError("Unsupported interpolation '" + value + "'", "$.input.interpolation");
}

void requirePositive(int value, const std::string& path) {
    if (value <= 0) throw ManifestError("Value must be greater than zero", path);
}

void requireNonNegative(int value, const std::string& path) {
    if (value < 0) throw ManifestError("Value must not be negative", path);
}

void requireUnit(float value, const std::string& path) {
    if (!std::isfinite(value) || value < 0.0F || value > 1.0F) {
        throw ManifestError("Value must be finite and in [0, 1]", path);
    }
}

ModelManifest parseManifest(const json::Value& value) {
    const Reader root(value);
    ModelManifest manifest;
    manifest.schemaVersion = root.required("schema_version").integer();
    if (manifest.schemaVersion != 1) {
        throw ManifestError("Only manifest schema_version 1 is supported", "$.schema_version");
    }

    const Reader model = root.required("model");
    manifest.modelId = model.required("id").string();
    manifest.modelVersion = model.required("version").string();
    manifest.algorithm = parseAlgorithm(model.required("algorithm").string());
    if (manifest.modelId.empty() || manifest.modelVersion.empty()) {
        throw ManifestError("Model id and version must not be empty", "$.model");
    }

    manifest.graphContract = parseContract(root.required("graph_contract").string());

    const Reader input = root.required("input");
    manifest.input.tensorName = optionalOr<std::string>(input, "tensor", "input", &Reader::string);
    manifest.input.layout = parseLayout(optionalOr<std::string>(input, "layout", "NCHW", &Reader::string));
    manifest.input.color = optionalOr<std::string>(input, "color", "RGB", &Reader::string);
    manifest.input.interpolation = parseInterpolation(
        optionalOr<std::string>(input, "interpolation", "linear", &Reader::string));
    if (manifest.input.color != "RGB" && manifest.input.color != "BGR") {
        throw ManifestError("Only RGB and BGR color orders are supported", "$.input.color");
    }
    std::array<int, 2> size{};
    if (const auto sizeNode = input.optional("size")) size = sizeNode->intPair();
    else if (const auto resizeNode = input.optional("resize")) size = resizeNode->intPair();
    else throw ManifestError("Missing required input size", "$.input.size");
    manifest.input.resizeHeight = size[0];
    manifest.input.resizeWidth = size[1];
    requirePositive(size[0], "$.input.size[0]");
    requirePositive(size[1], "$.input.size[1]");
    if (const auto crop = input.optional("center_crop")) {
        const auto cropSize = crop->intPair();
        requirePositive(cropSize[0], "$.input.center_crop[0]");
        requirePositive(cropSize[1], "$.input.center_crop[1]");
        if (cropSize[0] > size[0] || cropSize[1] > size[1]) {
            throw ManifestError("center_crop must fit inside resized image", "$.input.center_crop");
        }
        manifest.input.centerCrop = cv::Size(cropSize[1], cropSize[0]);
    }
    if (const auto mean = input.optional("mean")) manifest.input.mean = mean->floatTriple();
    if (const auto std = input.optional("std")) manifest.input.std = std->floatTriple();
    for (std::size_t i = 0; i < manifest.input.std.size(); ++i) {
        if (!std::isfinite(manifest.input.std[i]) || manifest.input.std[i] <= 0.0F) {
            throw ManifestError("Standard deviation must be finite and positive",
                                "$.input.std[" + std::to_string(i) + "]");
        }
    }

    const Reader runtime = root.required("runtime");
    manifest.runtime.backend = parseRuntimeBackend(
        optionalOr<std::string>(runtime, "backend", "tensorrt", &Reader::string));
    manifest.runtime.onnxFile = optionalPath(runtime, "onnx");
    manifest.runtime.engineFile = optionalPath(runtime, "engine");
    manifest.runtime.loadPolicy = parseLoadPolicy(
        optionalOr<std::string>(runtime, "load_policy", "prefer_engine", &Reader::string));
    manifest.runtime.fp16 = optionalOr<bool>(runtime, "fp16", false, &Reader::boolean);
    manifest.runtime.maxBatchSize = optionalOr<int>(runtime, "max_batch_size", 1, &Reader::integer);
    requirePositive(manifest.runtime.maxBatchSize, "$.runtime.max_batch_size");
    if (const auto workspace = runtime.optional("workspace_mb")) {
        const int megabytes = workspace->integer();
        requirePositive(megabytes, "$.runtime.workspace_mb");
        manifest.runtime.workspaceBytes = static_cast<std::size_t>(megabytes) * 1024ULL * 1024ULL;
    }

    if (runtime.optional("provider")) {
        throw ManifestError("runtime.provider is not a model property; select CPU/GPU through load options",
                            "$.runtime.provider");
    }
    if (runtime.optional("strict_provider")) {
        throw ManifestError("runtime.strict_provider is no longer supported",
                            "$.runtime.strict_provider");
    }
    manifest.runtime.deviceId = optionalOr<int>(runtime, "device_id", 0, &Reader::integer);
    manifest.runtime.intraOpThreads = optionalOr<int>(
        runtime, "intra_op_threads", 0, &Reader::integer);
    manifest.runtime.interOpThreads = optionalOr<int>(
        runtime, "inter_op_threads", 0, &Reader::integer);
    requireNonNegative(manifest.runtime.deviceId, "$.runtime.device_id");
    requireNonNegative(manifest.runtime.intraOpThreads, "$.runtime.intra_op_threads");
    requireNonNegative(manifest.runtime.interOpThreads, "$.runtime.inter_op_threads");
    manifest.runtime.ortGraphOptimization = parseOrtGraphOptimization(
        optionalOr<std::string>(runtime, "graph_optimization", "all", &Reader::string));
    manifest.runtime.ortExecutionMode = parseOrtExecutionMode(
        optionalOr<std::string>(runtime, "execution_mode", "sequential", &Reader::string));
    manifest.runtime.enableMemoryPattern = optionalOr<bool>(
        runtime, "enable_memory_pattern", true, &Reader::boolean);
    manifest.runtime.enableCpuMemoryArena = optionalOr<bool>(
        runtime, "enable_cpu_memory_arena", true, &Reader::boolean);
    manifest.runtime.enableProfiling = optionalOr<bool>(
        runtime, "enable_profiling", false, &Reader::boolean);
    manifest.runtime.profileFilePrefix = optionalPath(runtime, "profile_file_prefix");
    if (manifest.runtime.profileFilePrefix.empty()) {
        manifest.runtime.profileFilePrefix = "anom_ort_profile";
    }
    if (manifest.runtime.engineFile.empty() && manifest.runtime.onnxFile.empty()) {
        throw ManifestError("At least one of runtime.engine or runtime.onnx is required", "$.runtime");
    }
    if (manifest.runtime.loadPolicy == EngineLoadPolicy::EngineOnly && manifest.runtime.engineFile.empty()) {
        throw ManifestError("engine_only requires runtime.engine", "$.runtime.load_policy");
    }
    if (manifest.runtime.backend == RuntimeBackend::OnnxRuntime && manifest.runtime.onnxFile.empty()) {
        throw ManifestError("onnxruntime backend requires runtime.onnx", "$.runtime.onnx");
    }

    const Reader outputs = root.required("outputs");
    for (const auto& [semantic, tensor] : outputs.object()) {
        manifest.outputs.emplace(semantic, Reader(tensor, "$.outputs." + semantic).string());
    }
    if (manifest.outputs.empty()) throw ManifestError("At least one output binding is required", "$.outputs");

    if (const auto checksums = root.optional("checksums")) {
        for (const auto& [artifact, checksumValue] : checksums->object()) {
            std::string checksum = Reader(checksumValue, "$.checksums." + artifact).string();
            std::transform(checksum.begin(), checksum.end(), checksum.begin(),
                           [](unsigned char ch) { return static_cast<char>(std::tolower(ch)); });
            const bool valid = checksum.size() == 64 && std::all_of(checksum.begin(), checksum.end(),
                [](unsigned char ch) { return std::isdigit(ch) || (ch >= 'a' && ch <= 'f'); });
            if (!valid) throw ManifestError("Checksum must be a 64-character SHA-256 hex string",
                                            "$.checksums." + artifact);
            manifest.checksums.emplace(artifact, std::move(checksum));
        }
    }

    if (const auto post = root.optional("postprocess")) {
        manifest.postprocess.normalize = optionalOr<bool>(*post, "normalize", true, &Reader::boolean);
        manifest.postprocess.threshold = optionalOr<bool>(*post, "threshold", true, &Reader::boolean);
        manifest.postprocess.imageThreshold = optionalOr<float>(*post, "image_threshold", 0.5F, &Reader::real);
        manifest.postprocess.pixelThreshold = optionalOr<float>(*post, "pixel_threshold", 0.5F, &Reader::real);
        manifest.postprocess.imageMin = optionalOr<float>(*post, "image_min", 0.0F, &Reader::real);
        manifest.postprocess.imageMax = optionalOr<float>(*post, "image_max", 1.0F, &Reader::real);
        manifest.postprocess.pixelMin = optionalOr<float>(*post, "pixel_min", 0.0F, &Reader::real);
        manifest.postprocess.pixelMax = optionalOr<float>(*post, "pixel_max", 1.0F, &Reader::real);
        manifest.postprocess.imageSensitivity = optionalOr<float>(*post, "image_sensitivity", 0.5F, &Reader::real);
        manifest.postprocess.pixelSensitivity = optionalOr<float>(*post, "pixel_sensitivity", 0.5F, &Reader::real);
        manifest.postprocess.smooth = optionalOr<bool>(*post, "smooth", false, &Reader::boolean);
        manifest.postprocess.smoothKernel = optionalOr<int>(*post, "smooth_kernel", 3, &Reader::integer);
        manifest.postprocess.smoothSigma = optionalOr<float>(*post, "smooth_sigma", 0.0F, &Reader::real);
        manifest.postprocess.morphology = optionalOr<bool>(*post, "morphology", false, &Reader::boolean);
        manifest.postprocess.morphologyKernel = optionalOr<int>(*post, "morphology_kernel", 5, &Reader::integer);
        manifest.postprocess.morphologyIterations = optionalOr<int>(*post, "morphology_iterations", 1, &Reader::integer);
        manifest.postprocess.morphologyMode = optionalOr<std::string>(*post, "morphology_mode", "open", &Reader::string);
        manifest.postprocess.analyze = optionalOr<bool>(*post, "analyze", false, &Reader::boolean);
        manifest.postprocess.minRegionArea = optionalOr<int>(*post, "min_region_area", 16, &Reader::integer);
        manifest.postprocess.maxRegions = optionalOr<int>(*post, "max_regions", 0, &Reader::integer);
    }
    requireUnit(manifest.postprocess.imageSensitivity, "$.postprocess.image_sensitivity");
    requireUnit(manifest.postprocess.pixelSensitivity, "$.postprocess.pixel_sensitivity");
    if (manifest.postprocess.normalize &&
        (manifest.postprocess.imageMax <= manifest.postprocess.imageMin ||
         manifest.postprocess.pixelMax <= manifest.postprocess.pixelMin)) {
        throw ManifestError("Normalization max values must be greater than min values", "$.postprocess");
    }
    if (manifest.postprocess.smoothKernel <= 0 || manifest.postprocess.smoothKernel % 2 == 0) {
        throw ManifestError("smooth_kernel must be a positive odd integer", "$.postprocess.smooth_kernel");
    }
    if (!std::isfinite(manifest.postprocess.smoothSigma) || manifest.postprocess.smoothSigma < 0.0F) {
        throw ManifestError("smooth_sigma must be finite and non-negative", "$.postprocess.smooth_sigma");
    }
    if (manifest.postprocess.morphologyKernel <= 0 || manifest.postprocess.morphologyKernel % 2 == 0) {
        throw ManifestError("morphology_kernel must be a positive odd integer", "$.postprocess.morphology_kernel");
    }
    requireNonNegative(manifest.postprocess.morphologyIterations, "$.postprocess.morphology_iterations");
    if (manifest.postprocess.morphologyMode != "open" && manifest.postprocess.morphologyMode != "close") {
        throw ManifestError("morphology_mode must be 'open' or 'close'", "$.postprocess.morphology_mode");
    }
    requireNonNegative(manifest.postprocess.minRegionArea, "$.postprocess.min_region_area");
    requireNonNegative(manifest.postprocess.maxRegions, "$.postprocess.max_regions");

    const Reader algorithm = root.required("algorithm");
    const std::string algorithmType = algorithm.required("type").string();
    if (parseAlgorithm(algorithmType) != manifest.algorithm) {
        throw ManifestError("$.algorithm.type must match $.model.algorithm", "$.algorithm.type");
    }

    if (manifest.algorithm == AlgorithmType::PatchCore) {
        PatchCoreConfig config;
        config.indexFile = algorithm.required("index").string();
        config.featureLayers = algorithm.required("feature_layers").stringArray();
        config.embeddingDimension = algorithm.required("embedding_dimension").integer();
        config.numNeighbors = optionalOr<int>(algorithm, "num_neighbors", 1, &Reader::integer);
        config.poolingKernel = optionalOr<int>(algorithm, "pooling_kernel", 3, &Reader::integer);
        config.poolingStride = optionalOr<int>(algorithm, "pooling_stride", 1, &Reader::integer);
        config.poolingPadding = optionalOr<int>(algorithm, "pooling_padding", 1, &Reader::integer);
        config.gaussianSigma = optionalOr<float>(algorithm, "gaussian_sigma", 4.0F, &Reader::real);
        config.sqrtDistances = optionalOr<bool>(algorithm, "sqrt_distances", true, &Reader::boolean);
        config.weightedImageScore = optionalOr<bool>(algorithm, "weighted_image_score", true, &Reader::boolean);
        requirePositive(config.embeddingDimension, "$.algorithm.embedding_dimension");
        requirePositive(config.numNeighbors, "$.algorithm.num_neighbors");
        requirePositive(config.poolingKernel, "$.algorithm.pooling_kernel");
        requirePositive(config.poolingStride, "$.algorithm.pooling_stride");
        requireNonNegative(config.poolingPadding, "$.algorithm.pooling_padding");
        if (!std::isfinite(config.gaussianSigma) || config.gaussianSigma < 0.0F) {
            throw ManifestError("Expected a finite non-negative value", "$.algorithm.gaussian_sigma");
        }
        if (config.featureLayers.empty()) throw ManifestError("feature_layers must not be empty", "$.algorithm.feature_layers");
        manifest.algorithmConfig = std::move(config);
    } else if (manifest.algorithm == AlgorithmType::Padim) {
        PadimConfig config;
        config.statisticsFile = algorithm.required("statistics").string();
        config.channelIndicesFile = algorithm.required("channel_indices").string();
        config.featureLayers = algorithm.required("feature_layers").stringArray();
        config.embeddingDimension = algorithm.required("embedding_dimension").integer();
        const auto featureSize = algorithm.required("feature_map_size").intPair();
        config.featureHeight = featureSize[0];
        config.featureWidth = featureSize[1];
        config.gaussianSigma = optionalOr<float>(algorithm, "gaussian_sigma", 4.0F, &Reader::real);
        requirePositive(config.embeddingDimension, "$.algorithm.embedding_dimension");
        requirePositive(config.featureHeight, "$.algorithm.feature_map_size[0]");
        requirePositive(config.featureWidth, "$.algorithm.feature_map_size[1]");
        if (!std::isfinite(config.gaussianSigma) || config.gaussianSigma < 0.0F) {
            throw ManifestError("Expected a finite non-negative value", "$.algorithm.gaussian_sigma");
        }
        if (config.featureLayers.empty()) throw ManifestError("feature_layers must not be empty", "$.algorithm.feature_layers");
        manifest.algorithmConfig = std::move(config);
    } else if (manifest.algorithm == AlgorithmType::EfficientAD) {
        EfficientADConfig config;
        config.teacherSemantic = optionalOr<std::string>(
            algorithm, "teacher_semantic", "teacher", &Reader::string);
        config.studentSemantic = optionalOr<std::string>(
            algorithm, "student_semantic", "student", &Reader::string);
        config.reconstructionErrorSemantic = optionalOr<std::string>(
            algorithm, "reconstruction_error_semantic", "reconstruction_error", &Reader::string);
        config.hardWeight = optionalOr<float>(algorithm, "hard_weight", 1.0F, &Reader::real);
        config.softWeight = optionalOr<float>(algorithm, "soft_weight", 1.0F, &Reader::real);
        config.gaussianSigma = optionalOr<float>(algorithm, "gaussian_sigma", 4.0F, &Reader::real);
        if (!std::isfinite(config.hardWeight) || config.hardWeight < 0.0F ||
            !std::isfinite(config.softWeight) || config.softWeight < 0.0F ||
            config.hardWeight + config.softWeight <= 0.0F) {
            throw ManifestError("EfficientAD hard and soft weights must be finite, non-negative, "
                                "and not both zero", "$.algorithm");
        }
        if (!std::isfinite(config.gaussianSigma) || config.gaussianSigma < 0.0F) {
            throw ManifestError("Expected a finite non-negative value", "$.algorithm.gaussian_sigma");
        }
        manifest.algorithmConfig = std::move(config);
    } else if (manifest.algorithm == AlgorithmType::DFKDE) {
        DFKDEConfig config;
        config.statisticsFile = algorithm.required("statistics").string();
        config.featureLayers = algorithm.required("feature_layers").stringArray();
        config.embeddingDimension = algorithm.required("embedding_dimension").integer();
        config.nComponents = optionalOr<int>(algorithm, "n_components", 16, &Reader::integer);
        config.kernelSigma = optionalOr<float>(algorithm, "kernel_sigma", 10.0F, &Reader::real);
        config.gaussianSigma = optionalOr<float>(algorithm, "gaussian_sigma", 4.0F, &Reader::real);
        requirePositive(config.embeddingDimension, "$.algorithm.embedding_dimension");
        requirePositive(config.nComponents, "$.algorithm.n_components");
        if (config.nComponents > config.embeddingDimension) {
            throw ManifestError("n_components must not exceed embedding_dimension",
                                "$.algorithm.n_components");
        }
        if (!std::isfinite(config.kernelSigma) || config.kernelSigma <= 0.0F) {
            throw ManifestError("Expected a finite positive value", "$.algorithm.kernel_sigma");
        }
        if (!std::isfinite(config.gaussianSigma) || config.gaussianSigma < 0.0F) {
            throw ManifestError("Expected a finite non-negative value", "$.algorithm.gaussian_sigma");
        }
        if (config.featureLayers.empty()) throw ManifestError("feature_layers must not be empty", "$.algorithm.feature_layers");
        manifest.algorithmConfig = std::move(config);
    } else if (manifest.algorithm == AlgorithmType::SPADE) {
        SPADEConfig config;
        config.featureLayers = algorithm.required("feature_layers").stringArray();
        config.indexFiles = algorithm.required("indexes").stringArray();
        config.numNeighbors = optionalOr<int>(algorithm, "num_neighbors", 5, &Reader::integer);
        config.gaussianSigma = optionalOr<float>(algorithm, "gaussian_sigma", 4.0F, &Reader::real);
        requirePositive(config.numNeighbors, "$.algorithm.num_neighbors");
        if (!std::isfinite(config.gaussianSigma) || config.gaussianSigma < 0.0F) {
            throw ManifestError("Expected a finite non-negative value", "$.algorithm.gaussian_sigma");
        }
        if (config.featureLayers.empty()) throw ManifestError("feature_layers must not be empty", "$.algorithm.feature_layers");
        if (config.indexFiles.size() != config.featureLayers.size()) {
            throw ManifestError("indexes must align one-to-one with feature_layers",
                                "$.algorithm.indexes");
        }
        manifest.algorithmConfig = std::move(config);
    } else if (manifest.algorithm == AlgorithmType::Yolo) {
        YoloConfig config;
        config.detectionSemantic = optionalOr<std::string>(algorithm, "detection_semantic", "output", &Reader::string);
        config.numClasses = algorithm.required("num_classes").integer();
        config.confThreshold = optionalOr<float>(algorithm, "conf_threshold", 0.25F, &Reader::real);
        config.nmsThreshold = optionalOr<float>(algorithm, "nms_threshold", 0.45F, &Reader::real);
        requirePositive(config.numClasses, "$.algorithm.num_classes");
        requireUnit(config.confThreshold, "$.algorithm.conf_threshold");
        requireUnit(config.nmsThreshold, "$.algorithm.nms_threshold");
        if (config.detectionSemantic.empty()) {
            throw ManifestError("detection_semantic must not be empty", "$.algorithm.detection_semantic");
        }
        manifest.algorithmConfig = std::move(config);
    } else {
        DirectConfig config;
        config.scoreSemantic = optionalOr<std::string>(algorithm, "score_semantic", "pred_score", &Reader::string);
        config.mapSemantic = optionalOr<std::string>(algorithm, "map_semantic", "anomaly_map", &Reader::string);
        manifest.algorithmConfig = std::move(config);
    }

    if (manifest.graphContract == GraphContract::Prediction &&
        manifest.algorithm != AlgorithmType::Direct &&
        manifest.algorithm != AlgorithmType::EfficientAD &&
        manifest.algorithm != AlgorithmType::Yolo) {
        throw ManifestError("prediction graph contract must use the direct, efficientad or yolo adapter",
                            "$.graph_contract");
    }
    if (manifest.graphContract == GraphContract::FeaturePyramid &&
        manifest.algorithm != AlgorithmType::PatchCore &&
        manifest.algorithm != AlgorithmType::Padim &&
        manifest.algorithm != AlgorithmType::DFKDE &&
        manifest.algorithm != AlgorithmType::SPADE) {
        throw ManifestError("feature_pyramid graph contract requires patchcore, padim, dfkde or spade",
                            "$.graph_contract");
    }
    return manifest;
}

bool pathEscapes(const std::filesystem::path& relative) {
    if (relative.empty() || relative.is_absolute()) return true;
    const auto normalized = relative.lexically_normal();
    for (const auto& part : normalized) {
        if (part == "..") return true;
    }
    return false;
}

}  // namespace

Result<ModelPackage> ModelPackage::load(const std::filesystem::path& packagePath) {
    try {
        ModelPackage package;
        if (std::filesystem::is_directory(packagePath)) {
            package.root_ = std::filesystem::weakly_canonical(packagePath);
            package.manifestPath_ = package.root_ / "manifest.json";
        } else {
            package.manifestPath_ = std::filesystem::weakly_canonical(packagePath);
            package.root_ = package.manifestPath_.parent_path();
        }
        if (!std::filesystem::is_regular_file(package.manifestPath_)) {
            return Status::error(ErrorCode::ArtifactMissing, "Model manifest does not exist",
                                 package.manifestPath_.string());
        }
        std::ifstream stream(package.manifestPath_, std::ios::binary);
        if (!stream) {
            return Status::error(ErrorCode::IoError, "Unable to open model manifest",
                                 package.manifestPath_.string());
        }
        const std::string text((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
        auto document = json::parse(text);
        if (!document) return document.status();
        package.manifest_ = parseManifest(document.value());
        for (const auto& [artifact, expected] : package.manifest_.checksums) {
            auto artifactPath = package.resolveArtifact(artifact, true);
            if (!artifactPath) return artifactPath.status();
            auto actual = sha256File(artifactPath.value());
            if (!actual) return actual.status();
            if (actual.value() != expected) {
                return Status::error(ErrorCode::ArtifactCorrupt,
                                     "Model artifact SHA-256 does not match manifest",
                                     artifact);
            }
        }
        return package;
    } catch (const ManifestError& error) {
        return Status::error(ErrorCode::InvalidManifest, error.what(), error.path);
    } catch (const std::filesystem::filesystem_error& error) {
        return Status::error(ErrorCode::IoError, error.what(), packagePath.string());
    } catch (const std::exception& error) {
        return Status::error(ErrorCode::InvalidManifest, error.what(), packagePath.string());
    }
}

Result<std::filesystem::path> ModelPackage::resolveArtifact(
    const std::filesystem::path& relativePath, bool mustExist) const {
    if (pathEscapes(relativePath)) {
        return Status::error(ErrorCode::ArtifactPathEscape,
                             "Artifact path must be a non-empty relative path inside the model package",
                             relativePath.string());
    }
    try {
        const auto resolved = std::filesystem::weakly_canonical(root_ / relativePath);
        const auto relative = std::filesystem::relative(resolved, root_);
        if (pathEscapes(relative)) {
            return Status::error(ErrorCode::ArtifactPathEscape,
                                 "Resolved artifact escapes the model package", resolved.string());
        }
        if (mustExist && !std::filesystem::is_regular_file(resolved)) {
            return Status::error(ErrorCode::ArtifactMissing, "Model artifact does not exist", resolved.string());
        }
        return resolved;
    } catch (const std::filesystem::filesystem_error& error) {
        return Status::error(ErrorCode::IoError, error.what(), relativePath.string());
    }
}

}  // namespace anom::model
