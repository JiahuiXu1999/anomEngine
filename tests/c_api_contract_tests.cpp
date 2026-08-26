#include "anomEngine/anomEngine.h"

#include "infrastructure/utf8_path.h"

#include <cmath>
#include <cstdint>
#include <filesystem>
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
    anom_session_options_t options{};
    options.struct_size = sizeof(options);

    anom_session_t* session = nullptr;
    const std::string packageUtf8 = anom::model::pathToUtf8(package);
    const anom_status_t created =
        anom_session_create(packageUtf8.c_str(), &options, &session);
    require(created == ANOM_STATUS_OK, "C API session creation failed: " + lastError());
    require(session != nullptr, "C API returned a null session");

    anom_model_info_t info{};
    info.struct_size = sizeof(info);
    require(anom_session_get_model_info(session, &info) == ANOM_STATUS_OK,
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
    const anom_status_t predicted = anom_session_predict(session, &image, &prediction);
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
    anom_session_destroy(session);
}

void testInvalidArguments() {
    anom_session_t* session = reinterpret_cast<anom_session_t*>(1);
    const anom_status_t status = anom_session_create(nullptr, nullptr, &session);
    require(status == ANOM_STATUS_INVALID_ARGUMENT,
            "C API did not reject a null model package");
    require(session == nullptr, "C API did not clear output for an invalid call");
    require(!lastError().empty(), "C API did not report an error message");
}

}  // namespace

int main() {
    try {
        testPublicCAbi();
        testInvalidArguments();
        std::cout << "C API contract tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "C API contract tests failed: " << error.what() << '\n';
        return 1;
    }
}
