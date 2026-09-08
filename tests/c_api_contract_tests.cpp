#include "anomEngine/anomEngine.h"

#include "infrastructure/utf8_path.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

std::string lastError() {
    const std::size_t size = anom_get_last_error(nullptr, 0);
    std::vector<char> buffer(size);
    anom_get_last_error(buffer.data(), buffer.size());
    return buffer.data();
}

void testPublicCAbi() {
    require(anom_get_abi_version() == ANOM_ENGINE_ABI_VERSION,
            "Public ABI version is inconsistent");

    const std::filesystem::path package =
        std::filesystem::path(ANOM_TEST_DATA_DIR) / "direct_ort_package";
    anom_algorithm_options_t options{};
    options.struct_size = sizeof(options);
    options.device = ANOM_DEVICE_CPU;
    options.device_id = -1;
    options.precision = ANOM_PRECISION_FP32;

    anom_direct_t direct{};
    direct.struct_size = sizeof(direct);
    require(anom_direct_init(&direct) == ANOM_STATUS_OK, "Direct init failed");
    require(direct.abi_version == ANOM_ENGINE_ABI_VERSION, "Object ABI is wrong");
    const std::string packageUtf8 = anom::model::pathToUtf8(package);
    const anom_status_t created =
        direct.load(&direct, packageUtf8.c_str(), &options);
    require(created == ANOM_STATUS_OK, "Direct load failed: " + lastError());
    require(direct.internal != nullptr, "Direct load returned an empty object");

    anom_model_info_t info{};
    info.struct_size = sizeof(info);
    require(direct.get_model_info(&direct, &info) == ANOM_STATUS_OK,
            "C API model info query failed: " + lastError());
    require(std::string(info.model_id_utf8) == "ort-direct-test", "C API model id is wrong");
    require(std::string(info.algorithm_utf8) == "direct", "C API algorithm is wrong");
    require(std::string(info.backend_utf8) == "onnxruntime", "C API backend is wrong");

    const std::uint8_t pixels[12] = {
        0, 0, 0, 255, 0, 0,
        0, 255, 0, 0, 0, 255};
    anom_image_t image{};
    image.struct_size = sizeof(image);
    image.data = pixels;
    image.width = 2;
    image.height = 2;
    image.stride_bytes = 6;
    image.pixel_format = ANOM_PIXEL_FORMAT_RGB8;

    anom_prediction_t prediction{};
    prediction.struct_size = sizeof(prediction);
    const anom_status_t predicted = direct.predict(&direct, &image, &prediction);
    require(predicted == ANOM_STATUS_OK, "C API prediction failed: " + lastError());
    require(std::isfinite(prediction.score), "C API prediction score is not finite");
    require(prediction.map_width == 2 && prediction.map_height == 2,
            "C API anomaly map dimensions are wrong");
    require(prediction.anomaly_map != nullptr, "C API anomaly map is missing");
    require(std::string(prediction.model_id_utf8) == "ort-direct-test",
            "C API prediction model id is wrong");

    anom_prediction_release(&prediction);
    require(prediction.internal == nullptr && prediction.struct_size == sizeof(prediction),
            "C API prediction release did not reset the result");
    direct.release(&direct);
    require(direct.internal == nullptr, "Direct release did not reset the object");
    require(direct.load != nullptr && direct.predict != nullptr,
            "Release discarded the object's method table");
    require(direct.load(&direct, packageUtf8.c_str(), &options) == ANOM_STATUS_OK,
            "Released object could not be reused");
    direct.release(&direct);

    anom_yolo_t yolo{};
    yolo.struct_size = sizeof(yolo);
    require(anom_yolo_init(&yolo) == ANOM_STATUS_OK, "YOLO init failed");
    require(yolo.load(&yolo, packageUtf8.c_str(), &options) ==
                ANOM_STATUS_ALGORITHM_MISMATCH,
            "YOLO object accepted a Direct model package");
    require(yolo.internal == nullptr, "Rejected algorithm load retained state");
}

void testExecutionSelectionCAbi() {
    const std::filesystem::path package =
        std::filesystem::path(ANOM_TEST_DATA_DIR) / "direct_ort_package";
    const std::string packageUtf8 = anom::model::pathToUtf8(package);

    anom_algorithm_options_t options{};
    options.struct_size = sizeof(options);
    options.device = ANOM_DEVICE_CPU;
    options.device_id = -1;
    options.fallback = ANOM_FALLBACK_NONE;
    options.precision = ANOM_PRECISION_FP32;

    anom_direct_t direct{};
    direct.struct_size = sizeof(direct);
    require(anom_direct_init(&direct) == ANOM_STATUS_OK,
            "CPU Direct init failed: " + lastError());
    require(direct.load(&direct, packageUtf8.c_str(), &options) ==
                ANOM_STATUS_OK,
            "CPU Direct load failed: " + lastError());
    require(direct.internal != nullptr, "CPU Direct load returned no state");

    anom_execution_info_t execution{};
    execution.struct_size = sizeof(execution);
    require(direct.get_execution_info(&direct, &execution) == ANOM_STATUS_OK,
            "Execution info query failed: " + lastError());
    require(execution.requested_device == ANOM_DEVICE_CPU,
            "Execution info lost the requested CPU device");
    require(std::string(execution.backend_utf8) == "onnxruntime",
            "CPU request did not select ONNX Runtime");
    require(std::string(execution.execution_provider_utf8) == "cpu",
            "CPU request did not select the CPU provider");
    require(execution.device_id == -1 && execution.fallback_occurred == 0,
            "CPU execution info is inconsistent");
    require(execution.precision == ANOM_PRECISION_FP32,
            "CPU execution precision is wrong");
    direct.release(&direct);

    options.backend_utf8 = "tensorrt";
    require(direct.load(&direct, packageUtf8.c_str(), &options) ==
                ANOM_STATUS_INVALID_ARGUMENT,
            "CPU request unexpectedly accepted TensorRT");
    require(direct.internal == nullptr, "Rejected Direct load retained state");

    options.backend_utf8 = "onnxruntime";
    options.device = ANOM_DEVICE_GPU;
    // Force an unavailable device independently of whether CI has a GPU.
    options.device_id = INT32_MAX;
    options.fallback = ANOM_FALLBACK_LOAD_ONLY;
    options.precision = ANOM_PRECISION_AUTO;
    require(direct.load(&direct, packageUtf8.c_str(), &options) ==
                ANOM_STATUS_OK,
            "GPU-preferred Direct object did not fall back to CPU: " + lastError());
    execution = {};
    execution.struct_size = sizeof(execution);
    require(direct.get_execution_info(&direct, &execution) == ANOM_STATUS_OK,
            "Fallback execution info query failed: " + lastError());
    require(std::string(execution.execution_provider_utf8) == "cpu" &&
                execution.fallback_occurred == 1,
            "GPU-preferred session did not report CPU fallback");
    require(execution.fallback_reason_utf8 && *execution.fallback_reason_utf8,
            "GPU-preferred session did not report a fallback reason");
    direct.release(&direct);

    options.fallback = ANOM_FALLBACK_NONE;
    require(direct.load(&direct, packageUtf8.c_str(), &options) ==
                ANOM_STATUS_UNSUPPORTED,
            "Strict GPU request unexpectedly accepted an unavailable CUDA device");
    require(direct.internal == nullptr, "Rejected strict GPU load retained state");

    options.backend_utf8 = nullptr;
    options.device = ANOM_DEVICE_AUTO;
    options.fallback = ANOM_FALLBACK_NONE;
    require(direct.load(&direct, packageUtf8.c_str(), &options) ==
                ANOM_STATUS_OK,
            "AUTO Direct object did not select the available CPU runtime: " + lastError());
    execution = {};
    execution.struct_size = sizeof(execution);
    require(direct.get_execution_info(&direct, &execution) == ANOM_STATUS_OK,
            "AUTO execution info query failed: " + lastError());
    require(std::string(execution.execution_provider_utf8) == "cpu" &&
                execution.fallback_occurred == 1,
            "AUTO session did not report its CPU fallback");
    direct.release(&direct);
}

void testRuntimeProbeCAbi() {
    require(anom_runtime_probe("onnxruntime", "cpu", -1, nullptr) == ANOM_STATUS_OK,
            "CPU runtime probe failed: " + lastError());
    require(anom_runtime_probe("onnxruntime", "cuda", INT32_MAX, nullptr) == ANOM_STATUS_UNSUPPORTED,
            "An unavailable CUDA device passed probing");
    require(!lastError().empty(), "Unavailable device probe lost its diagnostic");
    require(anom_runtime_probe("onnxruntime", "cpu", -1, nullptr) == ANOM_STATUS_OK && lastError().empty(),
            "CPU probe did not clear the previous diagnostic");
    require(anom_runtime_probe("tensorrt", "cpu", 0, nullptr) == ANOM_STATUS_INVALID_ARGUMENT,
            "TensorRT CPU probe was accepted");
    require(anom_runtime_probe(nullptr, "cpu", 0, nullptr) == ANOM_STATUS_INVALID_ARGUMENT,
            "Null backend was accepted");
    require(anom_runtime_probe("onnxruntime", "invalid", 0, nullptr) == ANOM_STATUS_INVALID_ARGUMENT,
            "Invalid provider was accepted");
    require(anom_runtime_probe("onnxruntime", "cpu", -2, nullptr) == ANOM_STATUS_INVALID_ARGUMENT,
            "Invalid device id was accepted");
    const auto missing = std::filesystem::path(ANOM_TEST_DATA_DIR) / "missing_plugins";
    require(anom_runtime_probe("onnxruntime", "cpu", 0, anom::model::pathToUtf8(missing).c_str()) ==
                ANOM_STATUS_PLUGIN_NOT_FOUND, "Missing plugin was not reported");
}

void testIndependentAlgorithmObjects() {
    const auto package =
        std::filesystem::path(ANOM_TEST_DATA_DIR) / "direct_ort_package";
    const std::string packageUtf8 = anom::model::pathToUtf8(package);
    anom_algorithm_options_t options{};
    options.struct_size = sizeof(options);
    options.device = ANOM_DEVICE_CPU;
    options.device_id = -1;
    options.precision = ANOM_PRECISION_FP32;

    anom_direct_t first{};
    anom_direct_t second{};
    first.struct_size = sizeof(first);
    second.struct_size = sizeof(second);
    require(anom_direct_init(&first) == ANOM_STATUS_OK, "First object init failed");
    require(anom_direct_init(&second) == ANOM_STATUS_OK, "Second object init failed");
    require(first.load(&first, packageUtf8.c_str(), &options) == ANOM_STATUS_OK,
            "First object load failed: " + lastError());
    require(second.load(&second, packageUtf8.c_str(), &options) == ANOM_STATUS_OK,
            "Second object load failed: " + lastError());
    require(first.internal != second.internal, "Objects unexpectedly share their runtime");
    first.release(&first);
    require(second.warmup(&second) == ANOM_STATUS_OK,
            "Releasing the first object affected the second: " + lastError());
    second.release(&second);
}

void testModelManagementCAbi() {
    const std::filesystem::path package =
        std::filesystem::path(ANOM_TEST_DATA_DIR) / "direct_ort_package";
    const std::string packageUtf8 = anom::model::pathToUtf8(package);

    bool sawPatchCore = false;
    bool sawYolo = false;
    for (std::size_t index = 0; index < anom_algorithm_get_count(); ++index) {
        anom_algorithm_info_t algorithm{};
        algorithm.struct_size = sizeof(algorithm);
        require(anom_algorithm_get_info(index, &algorithm) == ANOM_STATUS_OK,
                "Algorithm capability query failed: " + lastError());
        if (std::string(algorithm.algorithm_utf8) == "patchcore") {
            sawPatchCore = true;
            require((algorithm.capabilities & ANOM_ALGORITHM_CAP_ARTIFACT_FIT) != 0,
                    "PatchCore did not advertise artifact fitting");
        }
        if (std::string(algorithm.algorithm_utf8) == "yolo") {
            sawYolo = true;
            require((algorithm.capabilities & ANOM_ALGORITHM_CAP_ARTIFACT_FIT) == 0,
                    "YOLO incorrectly advertised artifact fitting");
        }
    }
    require(sawPatchCore && sawYolo, "Expected algorithms are absent from capability discovery");

    anom_model_t* model = nullptr;
    require(anom_model_open(packageUtf8.c_str(), &model) == ANOM_STATUS_OK,
            "Model open failed: " + lastError());
    require(model != nullptr, "Model open returned null");
    anom_model_info_t info{};
    info.struct_size = sizeof(info);
    require(anom_model_get_info(model, &info) == ANOM_STATUS_OK,
            "Model info failed: " + lastError());
    require(std::string(info.algorithm_utf8) == "direct", "Opened model algorithm is wrong");

    anom_calibrator_options_t calibratorOptions{};
    calibratorOptions.struct_size = sizeof(calibratorOptions);
    calibratorOptions.target_false_positive_rate = 0.1F;
    calibratorOptions.max_pixel_samples = 32;
    anom_calibrator_t* calibrator = nullptr;
    require(anom_calibrator_create(model, &calibratorOptions, &calibrator) == ANOM_STATUS_OK,
            "Calibrator create failed: " + lastError());
    const std::uint8_t calibrationPixels[12] = {
        0, 0, 0, 32, 32, 32, 64, 64, 64, 255, 255, 255};
    anom_image_t calibrationImage{};
    calibrationImage.struct_size = sizeof(calibrationImage);
    calibrationImage.data = calibrationPixels;
    calibrationImage.width = 2;
    calibrationImage.height = 2;
    calibrationImage.stride_bytes = 6;
    calibrationImage.pixel_format = ANOM_PIXEL_FORMAT_RGB8;
    require(anom_calibrator_add_batch(calibrator, &calibrationImage, 1) == ANOM_STATUS_OK,
            "Calibrator add batch failed: " + lastError());
    anom_calibration_result_t calibration{};
    calibration.struct_size = sizeof(calibration);
    require(anom_calibrator_compute(calibrator, &calibration) == ANOM_STATUS_OK,
            "Calibration compute failed: " + lastError());
    require(calibration.sample_count == 1 && calibration.has_pixel_statistics == 1,
            "Calibration result is incomplete");
    anom_calibrator_destroy(calibrator);
    anom_model_destroy(model);

    anom_model_validation_report_t validation{};
    validation.struct_size = sizeof(validation);
    require(anom_model_validate(packageUtf8.c_str(), nullptr, &validation) == ANOM_STATUS_OK,
            "Static model validation failed: " + lastError());
    require(validation.package_valid == 1 && validation.runtime_valid == 0,
            "Static validation report is wrong");

    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto output = std::filesystem::temp_directory_path() /
                        ("anom-cabi-package-" + unique);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{output};

    anom_package_builder_options_t builderOptions{};
    builderOptions.struct_size = sizeof(builderOptions);
    builderOptions.template_package_utf8 = packageUtf8.c_str();
    anom_package_builder_t* builder = nullptr;
    require(anom_package_builder_create(&builderOptions, &builder) == ANOM_STATUS_OK,
            "Package builder create failed: " + lastError());
    require(anom_package_builder_apply_calibration(builder, &calibration) == ANOM_STATUS_OK,
            "Applying calibration to package builder failed: " + lastError());
    const std::string outputUtf8 = anom::model::pathToUtf8(output);
    require(anom_package_builder_commit(builder, outputUtf8.c_str()) == ANOM_STATUS_OK,
            "Package builder commit failed: " + lastError());
    anom_package_builder_destroy(builder);
    require(std::filesystem::is_regular_file(output / "manifest.json"),
            "Package builder did not write a manifest");
    require(anom_model_open(outputUtf8.c_str(), &model) == ANOM_STATUS_OK,
            "Committed package could not be opened: " + lastError());
    anom_model_destroy(model);

    anom_patchcore_fitter_options_t wrongOptions{};
    wrongOptions.struct_size = sizeof(wrongOptions);
    wrongOptions.template_package_utf8 = packageUtf8.c_str();
    anom_patchcore_fitter_t wrongFitter{};
    wrongFitter.struct_size = sizeof(wrongFitter);
    require(anom_patchcore_fitter_init(&wrongFitter) == ANOM_STATUS_OK,
            "PatchCore fitter init failed");
    require(wrongFitter.create(&wrongFitter, &wrongOptions) ==
                ANOM_STATUS_ALGORITHM_MISMATCH,
            "PatchCore fitter accepted a Direct model template");
    require(wrongFitter.internal == nullptr,
            "Rejected PatchCore fitter retained state");
}

void testPatchCoreFitterCAbi() {
    bool available = false;
    for (std::size_t index = 0; index < anom_algorithm_get_count(); ++index) {
        anom_algorithm_info_t info{};
        info.struct_size = sizeof(info);
        require(anom_algorithm_get_info(index, &info) == ANOM_STATUS_OK,
                "Capability query failed: " + lastError());
        if (std::string(info.algorithm_utf8) == "patchcore") available = info.available != 0;
    }
    if (!available) return;

    const auto unique = std::to_string(
        std::chrono::steady_clock::now().time_since_epoch().count());
    const auto root = std::filesystem::temp_directory_path() /
                      ("anom-cabi-fitter-" + unique);
    const auto templatePackage = root / "template";
    const auto outputPackage = root / "model";
    const auto checkpoint = root / "checkpoint.bin";
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() { std::error_code ignored; std::filesystem::remove_all(path, ignored); }
    } cleanup{root};
    std::filesystem::create_directories(templatePackage);
    std::filesystem::copy_file(
        std::filesystem::path(ANOM_TEST_DATA_DIR) / "identity_dynamic.onnx",
        templatePackage / "backbone.onnx");
    std::ofstream manifest(templatePackage / "manifest.json", std::ios::binary);
    manifest << R"JSON({
      "schema_version":1,
      "model":{"id":"cabi-patchcore","version":"1","algorithm":"patchcore"},
      "graph_contract":"feature_pyramid",
      "input":{"tensor":"input","layout":"NCHW","color":"RGB","size":[2,2],
        "mean":[0,0,0],"std":[1,1,1]},
      "runtime":{"backend":"onnxruntime","onnx":"backbone.onnx",
        "max_batch_size":2,"intra_op_threads":1,"inter_op_threads":1},
      "outputs":{"layer":"output"},
      "algorithm":{"type":"patchcore","index":"memory.faiss",
        "feature_layers":["layer"],"embedding_dimension":3,"num_neighbors":1,
        "pooling_kernel":1,"pooling_stride":1,"pooling_padding":0,
        "sqrt_distances":true,"weighted_image_score":false,"gaussian_sigma":0},
      "postprocess":{"normalize":false,"threshold":false}
    })JSON";
    manifest.close();

    const std::string templateUtf8 = anom::model::pathToUtf8(templatePackage);
    anom_patchcore_fitter_options_t options{};
    options.struct_size = sizeof(options);
    options.template_package_utf8 = templateUtf8.c_str();
    options.coreset_sampling_ratio = 0.5F;
    options.projection_dimension = 3;
    options.random_seed = 7;
    anom_patchcore_fitter_t fitter{};
    fitter.struct_size = sizeof(fitter);
    require(anom_patchcore_fitter_init(&fitter) == ANOM_STATUS_OK,
            "PatchCore fitter init failed");
    require(fitter.create(&fitter, &options) == ANOM_STATUS_OK,
            "PatchCore fitter create failed: " + lastError());

    const std::uint8_t pixels[24] = {
        0, 0, 0, 32, 32, 32, 64, 64, 64, 96, 96, 96,
        128, 128, 128, 160, 160, 160, 192, 192, 192, 255, 255, 255};
    anom_image_t images[2]{};
    for (std::size_t index = 0; index < 2; ++index) {
        images[index].struct_size = sizeof(anom_image_t);
        images[index].data = pixels + index * 12;
        images[index].width = 2;
        images[index].height = 2;
        images[index].stride_bytes = 6;
        images[index].pixel_format = ANOM_PIXEL_FORMAT_RGB8;
    }
    require(fitter.add_batch(&fitter, images, 2) == ANOM_STATUS_OK,
            "PatchCore fitter add batch failed: " + lastError());
    anom_fit_progress_t progress{};
    progress.struct_size = sizeof(progress);
    require(fitter.get_progress(&fitter, &progress) == ANOM_STATUS_OK,
            "PatchCore fitter progress failed: " + lastError());
    require(progress.processed_samples == 2 && progress.collected_items == 8,
            "PatchCore fitter reported incorrect progress");
    const std::string checkpointUtf8 = anom::model::pathToUtf8(checkpoint);
    require(fitter.save_checkpoint(&fitter, checkpointUtf8.c_str()) ==
                ANOM_STATUS_OK,
            "PatchCore checkpoint save failed: " + lastError());
    const std::string outputUtf8 = anom::model::pathToUtf8(outputPackage);
    require(fitter.finalize(&fitter, outputUtf8.c_str()) == ANOM_STATUS_OK,
            "PatchCore fitter finalize failed: " + lastError());
    fitter.release(&fitter);
    require(fitter.internal == nullptr, "PatchCore fitter release retained state");

    require(std::filesystem::is_regular_file(outputPackage / "memory.faiss"),
            "PatchCore fitter did not produce a memory bank");
    anom_model_validation_report_t report{};
    report.struct_size = sizeof(report);
    anom_model_validate_options_t validateOptions{};
    validateOptions.struct_size = sizeof(validateOptions);
    validateOptions.initialize_runtime = 1;
    require(anom_model_validate(outputUtf8.c_str(), &validateOptions, &report) == ANOM_STATUS_OK,
            "Fitted PatchCore package is invalid: " + lastError());
    require(report.package_valid == 1 && report.runtime_valid == 1,
            "Fitted PatchCore runtime validation report is incomplete");
}

void testInvalidArguments() {
    anom_direct_t direct{};
    direct.struct_size = sizeof(direct);
    require(anom_direct_init(&direct) == ANOM_STATUS_OK,
            "Direct init failed for invalid-argument test");
    const anom_status_t status = direct.load(&direct, nullptr, nullptr);
    require(status == ANOM_STATUS_INVALID_ARGUMENT,
            "C API did not reject a null model package");
    require(direct.internal == nullptr, "C API retained state for an invalid call");
    require(!lastError().empty(), "C API did not report an error message");
}

}  // namespace

int main() {
    try {
        testPublicCAbi();
        testExecutionSelectionCAbi();
        testRuntimeProbeCAbi();
        testIndependentAlgorithmObjects();
        testModelManagementCAbi();
        testPatchCoreFitterCAbi();
        testInvalidArguments();
        std::cout << "C API contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C API contract tests failed: " << error.what() << '\n';
        return 1;
    }
}
