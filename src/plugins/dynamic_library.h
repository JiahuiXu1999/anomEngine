#pragma once

#include "model/result.h"

#include <filesystem>
#include <memory>

namespace anom::model::plugins {

class DynamicLibrary {
public:
    static Result<std::shared_ptr<DynamicLibrary>> open(const std::filesystem::path& path);
    ~DynamicLibrary();

    DynamicLibrary(const DynamicLibrary&) = delete;
    DynamicLibrary& operator=(const DynamicLibrary&) = delete;

    [[nodiscard]] void* symbol(const char* name) const noexcept;
    [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

private:
    DynamicLibrary(std::filesystem::path path, void* handle)
        : path_(std::move(path)), handle_(handle) {}

    std::filesystem::path path_;
    void* handle_{nullptr};
};

[[nodiscard]] std::filesystem::path defaultPluginDirectory();
[[nodiscard]] std::filesystem::path pluginPath(
    const std::filesystem::path& directory,
    const char* category,
    const char* name);

}  // namespace anom::model::plugins
