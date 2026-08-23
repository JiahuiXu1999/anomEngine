#include "adapters/direct_adapter.h"

#include <cstring>

namespace anom::model {
namespace {

Result<std::string> bindingFor(const std::unordered_map<std::string, std::string>& bindings,
                               const std::string& semantic, bool required) {
    const auto found = bindings.find(semantic);
    if (found == bindings.end()) {
        if (!required) return std::string{};
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Required output semantic is not bound", semantic);
    }
    return found->second;
}

}  // namespace

Result<void> DirectPredictionAdapter::validateSignature(const TensorSignature& signature) const {
    auto scoreName = bindingFor(outputBindings_, config_.scoreSemantic, true);
    if (!scoreName) return scoreName.status();
    const auto* score = signature.findOutput(scoreName.value());
    if (!score) return Status::error(ErrorCode::TensorSignatureMismatch,
                                     "Score output is missing from backend", scoreName.value());
    if (score->dtype != DataType::Float32) {
        return Status::error(ErrorCode::TensorTypeMismatch,
                             "Direct score output must use float32", scoreName.value());
    }
    auto mapName = bindingFor(outputBindings_, config_.mapSemantic, false);
    if (mapName && !mapName.value().empty()) {
        const auto* map = signature.findOutput(mapName.value());
        if (!map) return Status::error(ErrorCode::TensorSignatureMismatch,
                                       "Anomaly map output is missing from backend", mapName.value());
        if (map->dtype != DataType::Float32) {
            return Status::error(ErrorCode::TensorTypeMismatch,
                                 "Direct anomaly map output must use float32", mapName.value());
        }
    }
    return {};
}

Result<RawPredictionBatch> DirectPredictionAdapter::predict(
    const TensorMap& outputs, const cv::Size&) const {
    auto scoreName = bindingFor(outputBindings_, config_.scoreSemantic, true);
    if (!scoreName) return scoreName.status();
    const auto scoreFound = outputs.find(scoreName.value());
    if (scoreFound == outputs.end()) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Backend did not return score output", scoreName.value());
    }
    const Tensor& scores = scoreFound->second;
    if (scores.dtype != DataType::Float32 || scores.shape.dims.empty()) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "Score output must be float32 and include a batch dimension", scoreName.value());
    }
    const int batch = static_cast<int>(scores.shape.dims[0]);
    auto count = scores.shape.elementCount();
    if (!count) return count.status();
    if (batch <= 0 || count.value() != static_cast<std::size_t>(batch)) {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "Score output must contain exactly one value per image", scores.shape.toString());
    }

    RawPredictionBatch result(static_cast<std::size_t>(batch));
    for (int n = 0; n < batch; ++n) result[static_cast<std::size_t>(n)].score = scores.data<float>()[n];

    auto mapName = bindingFor(outputBindings_, config_.mapSemantic, false);
    if (!mapName || mapName.value().empty()) return result;
    const auto mapFound = outputs.find(mapName.value());
    if (mapFound == outputs.end()) {
        return Status::error(ErrorCode::TensorSignatureMismatch,
                             "Backend did not return anomaly map output", mapName.value());
    }
    const Tensor& maps = mapFound->second;
    int height = 0, width = 0;
    if (maps.dtype != DataType::Float32) {
        return Status::error(ErrorCode::TensorTypeMismatch, "Anomaly map must use float32", mapName.value());
    }
    if (maps.shape.dims.size() == 4 && maps.shape.dims[1] == 1) {
        height = static_cast<int>(maps.shape.dims[2]); width = static_cast<int>(maps.shape.dims[3]);
    } else if (maps.shape.dims.size() == 3) {
        height = static_cast<int>(maps.shape.dims[1]); width = static_cast<int>(maps.shape.dims[2]);
    } else {
        return Status::error(ErrorCode::TensorShapeMismatch,
                             "Anomaly map must have shape [N,1,H,W] or [N,H,W]", maps.shape.toString());
    }
    if (maps.shape.dims[0] != batch || height <= 0 || width <= 0) {
        return Status::error(ErrorCode::TensorShapeMismatch, "Invalid anomaly map shape", maps.shape.toString());
    }
    const std::size_t mapElements = static_cast<std::size_t>(height) * width;
    for (int n = 0; n < batch; ++n) {
        cv::Mat map(height, width, CV_32FC1,
                    const_cast<float*>(maps.data<float>() + static_cast<std::size_t>(n) * mapElements));
        result[static_cast<std::size_t>(n)].anomalyMap = map.clone();
    }
    return result;
}

}  // namespace anom::model
