#include "adapters/adapter.h"

#include "adapters/dfkde_adapter.h"
#include "adapters/direct_adapter.h"
#include "adapters/efficientad_adapter.h"
#include "adapters/padim_adapter.h"
#include "adapters/patchcore_adapter.h"
#include "adapters/spade_adapter.h"

namespace anom::model {

Result<std::unique_ptr<IModelAdapter>> createAdapter(const ModelPackage& package) {
    const auto& manifest = package.manifest();
    std::unique_ptr<IModelAdapter> adapter;
    switch (manifest.algorithm) {
        case AlgorithmType::PatchCore:
            adapter = std::make_unique<PatchCoreAdapter>(
                std::get<PatchCoreConfig>(manifest.algorithmConfig), manifest.outputs);
            break;
        case AlgorithmType::Padim:
            adapter = std::make_unique<PadimAdapter>(
                std::get<PadimConfig>(manifest.algorithmConfig), manifest.outputs);
            break;
        case AlgorithmType::Direct:
            adapter = std::make_unique<DirectPredictionAdapter>(
                std::get<DirectConfig>(manifest.algorithmConfig), manifest.outputs);
            break;
        case AlgorithmType::EfficientAD:
            adapter = std::make_unique<EfficientADAdapter>(
                std::get<EfficientADConfig>(manifest.algorithmConfig), manifest.outputs);
            break;
        case AlgorithmType::DFKDE:
            adapter = std::make_unique<DfkdeAdapter>(
                std::get<DFKDEConfig>(manifest.algorithmConfig), manifest.outputs);
            break;
        case AlgorithmType::SPADE:
            adapter = std::make_unique<SpadeAdapter>(
                std::get<SPADEConfig>(manifest.algorithmConfig), manifest.outputs);
            break;
    }
    auto loaded = adapter->loadAssets(package);
    if (!loaded) return loaded.status();
    return adapter;
}

}  // namespace anom::model
