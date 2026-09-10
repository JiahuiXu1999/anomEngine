#pragma once
#include "model/runtime_types.h"

namespace anom::model {
struct SearchExecutionConfig {
    ExecutionProvider provider{ExecutionProvider::Cpu};
    int deviceId{0};
    bool allowCpuFallback{false};
    std::filesystem::path pluginDirectory;
};
struct SearchExecutionInfo {
    ExecutionProvider provider{ExecutionProvider::Default}; // Default = no Faiss stage
    int deviceId{-1};
    bool fallbackOccurred{false};
    std::string fallbackReason;
};
inline bool retryableSearchFailure(ErrorCode code) {
    return code == ErrorCode::DeviceUnavailable || code == ErrorCode::PluginNotFound ||
           code == ErrorCode::PluginAbiMismatch || code == ErrorCode::CudaFailure ||
           code == ErrorCode::OutOfMemory;
}
}
