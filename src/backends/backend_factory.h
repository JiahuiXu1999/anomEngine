#pragma once

#include "backends/backend.h"

namespace anom::model {

// Constructs the requested runtime without loading model artifacts.
// Plugin loading is checked here; probe() checks provider/device availability,
// and load() additionally validates the model and allocates execution resources.
Result<std::unique_ptr<IRuntimeBackend>> createRuntimeBackend(
    RuntimeBackend backend,
    const std::filesystem::path& pluginDirectory = {});

}  // namespace anom::model
