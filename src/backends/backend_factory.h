#pragma once

#include "backends/backend.h"

namespace anom::model {

// Constructs the requested runtime without loading model artifacts.
// Runtime availability is decided at build time and reported as a Status.
Result<std::unique_ptr<IRuntimeBackend>> createRuntimeBackend(RuntimeBackend backend);

}  // namespace anom::model
