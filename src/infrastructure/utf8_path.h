#ifndef ANOM_ENGINE_INFRASTRUCTURE_UTF8_PATH_H
#define ANOM_ENGINE_INFRASTRUCTURE_UTF8_PATH_H

#include <cstring>
#include <filesystem>
#include <string>
#include <string_view>

namespace anom::model {

// The public and plugin ABIs represent paths as UTF-8 char strings. C++20
// deliberately represents filesystem UTF-8 data with char8_t, so all boundary
// conversions live here instead of relying on platform-native narrow strings.
[[nodiscard]] inline std::string pathToUtf8(const std::filesystem::path& path) {
    const std::u8string encoded = path.u8string();
    std::string result(encoded.size(), '\0');
    if (!encoded.empty()) {
        std::memcpy(result.data(), encoded.data(), encoded.size());
    }
    return result;
}

[[nodiscard]] inline std::filesystem::path pathFromUtf8(std::string_view encoded) {
    std::u8string value(encoded.size(), u8'\0');
    if (!encoded.empty()) {
        std::memcpy(value.data(), encoded.data(), encoded.size());
    }
    return std::filesystem::path(value);
}

}  // namespace anom::model

#endif
