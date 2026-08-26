#include "plugins/dynamic_library.h"

#include "infrastructure/utf8_path.h"

#include <string>

#if defined(_WIN32)
#  define WIN32_LEAN_AND_MEAN
#  include <windows.h>
#else
#  include <dlfcn.h>
#endif

namespace anom::model::plugins {
namespace {

#if defined(_WIN32)
std::string windowsError(DWORD code) {
    wchar_t* text = nullptr;
    const DWORD length = FormatMessageW(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<wchar_t*>(&text), 0, nullptr);
    if (length == 0 || !text) return "Windows error " + std::to_string(code);
    const int utf8Length = WideCharToMultiByte(CP_UTF8, 0, text, static_cast<int>(length),
                                                nullptr, 0, nullptr, nullptr);
    std::string result(static_cast<std::size_t>(utf8Length), '\0');
    WideCharToMultiByte(CP_UTF8, 0, text, static_cast<int>(length), result.data(),
                        utf8Length, nullptr, nullptr);
    LocalFree(text);
    while (!result.empty() && (result.back() == '\r' || result.back() == '\n')) result.pop_back();
    return result;
}
#endif

}  // namespace

Result<std::shared_ptr<DynamicLibrary>> DynamicLibrary::open(
    const std::filesystem::path& path) {
    const auto absolute = std::filesystem::absolute(path).lexically_normal();
#if defined(_WIN32)
    HMODULE handle = LoadLibraryExW(
        absolute.c_str(), nullptr,
        LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS);
    if (!handle) {
        return Status::error(ErrorCode::IoError, "Unable to load plugin library",
                             pathToUtf8(absolute) + ": " + windowsError(GetLastError()));
    }
    return std::shared_ptr<DynamicLibrary>(
        new DynamicLibrary(absolute, reinterpret_cast<void*>(handle)));
#else
    void* handle = dlopen(absolute.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        const char* error = dlerror();
        return Status::error(ErrorCode::IoError, "Unable to load plugin library",
                             absolute.string() + ": " + (error ? error : "unknown error"));
    }
    return std::shared_ptr<DynamicLibrary>(new DynamicLibrary(absolute, handle));
#endif
}

DynamicLibrary::~DynamicLibrary() {
    if (!handle_) return;
#if defined(_WIN32)
    FreeLibrary(reinterpret_cast<HMODULE>(handle_));
#else
    dlclose(handle_);
#endif
}

void* DynamicLibrary::symbol(const char* name) const noexcept {
#if defined(_WIN32)
    return reinterpret_cast<void*>(GetProcAddress(reinterpret_cast<HMODULE>(handle_), name));
#else
    return dlsym(handle_, name);
#endif
}

std::filesystem::path defaultPluginDirectory() {
#if defined(_WIN32)
    HMODULE module = nullptr;
    if (!GetModuleHandleExW(
            GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS | GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
            reinterpret_cast<LPCWSTR>(&defaultPluginDirectory), &module)) {
        return std::filesystem::current_path();
    }
    std::wstring buffer(32768, L'\0');
    const DWORD length = GetModuleFileNameW(module, buffer.data(), static_cast<DWORD>(buffer.size()));
    if (length == 0 || length == buffer.size()) return std::filesystem::current_path();
    buffer.resize(length);
    return std::filesystem::path(buffer).parent_path();
#else
    Dl_info info{};
    if (dladdr(reinterpret_cast<void*>(&defaultPluginDirectory), &info) == 0 || !info.dli_fname) {
        return std::filesystem::current_path();
    }
    return std::filesystem::path(info.dli_fname).parent_path();
#endif
}

std::filesystem::path pluginPath(const std::filesystem::path& directory,
                                 const char* category, const char* name) {
#if defined(_WIN32)
    constexpr const char* extension = ".dll";
#elif defined(__APPLE__)
    constexpr const char* extension = ".dylib";
#else
    constexpr const char* extension = ".so";
#endif
    return directory / (std::string("anom_") + category + "_" + name + extension);
}

}  // namespace anom::model::plugins
