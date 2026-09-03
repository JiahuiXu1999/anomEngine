#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace anom::model {

enum class ErrorCode {
    Ok = 0,
    InvalidArgument,
    InvalidImage,
    IoError,
    InvalidManifest,
    UnsupportedSchema,
    UnsupportedAlgorithm,
    ArtifactMissing,
    ArtifactPathEscape,
    ArtifactCorrupt,
    TensorSignatureMismatch,
    TensorShapeMismatch,
    TensorTypeMismatch,
    EngineIncompatible,
    BackendFailure,
    AdapterFailure,
    CudaFailure,
    OutOfMemory,
    NotInitialized,
    InternalError,
    PluginNotFound,
    PluginAbiMismatch,
    DeviceUnavailable,
};

struct Status {
    ErrorCode code{ErrorCode::Ok};
    std::string message;
    std::string context;

    [[nodiscard]] bool ok() const noexcept { return code == ErrorCode::Ok; }

    static Status success() { return {}; }

    static Status error(ErrorCode code, std::string message, std::string context = {}) {
        return Status{code, std::move(message), std::move(context)};
    }

    [[nodiscard]] std::string describe() const {
        return context.empty() ? message : message + " [" + context + "]";
    }
};

template <typename T>
class Result {
public:
    Result(T value) : value_(std::move(value)), status_(Status::success()) {}
    Result(Status status) : status_(std::move(status)) {
        if (status_.ok()) {
            throw std::logic_error("A failed Result cannot contain an OK status");
        }
    }

    [[nodiscard]] bool ok() const noexcept { return value_.has_value(); }
    explicit operator bool() const noexcept { return ok(); }

    T& value() & {
        ensureValue();
        return *value_;
    }

    const T& value() const& {
        ensureValue();
        return *value_;
    }

    T&& value() && {
        ensureValue();
        return std::move(*value_);
    }

    [[nodiscard]] const Status& status() const noexcept { return status_; }

private:
    void ensureValue() const {
        if (!value_) {
            throw std::logic_error("Attempted to access a failed Result: " + status_.describe());
        }
    }

    std::optional<T> value_;
    Status status_;
};

template <>
class Result<void> {
public:
    Result() : status_(Status::success()) {}
    Result(Status status) : status_(std::move(status)) {}

    [[nodiscard]] bool ok() const noexcept { return status_.ok(); }
    explicit operator bool() const noexcept { return ok(); }
    [[nodiscard]] const Status& status() const noexcept { return status_; }

private:
    Status status_;
};

}  // namespace anom::model
