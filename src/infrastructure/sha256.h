#pragma once

#include "model/result.h"

#include <filesystem>
#include <string>

namespace anom::model {

Result<std::string> sha256File(const std::filesystem::path& path);

}  // namespace anom::model
