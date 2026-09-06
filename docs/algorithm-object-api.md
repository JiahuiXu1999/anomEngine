# Algorithm object API

The public ABI models every algorithm as a distinct object. There is no generic
inference session in the v2 header surface.

| Algorithm | C object | C++ facade |
|---|---|---|
| Direct Prediction | `anom_direct_t` | `anom::Direct` |
| EfficientAD | `anom_efficientad_t` | `anom::EfficientAD` |
| DFKDE | `anom_dfkde_t` | `anom::DFKDE` |
| PaDiM | `anom_padim_t` | `anom::PaDiM` |
| PatchCore | `anom_patchcore_t` | `anom::PatchCore` |
| SPADE | `anom_spade_t` | `anom::SPADE` |
| YOLO | `anom_yolo_t` | `anom::Yolo` |

All C objects are caller-owned, non-copyable values. Zero-initialize the object,
set `struct_size`, call its `*_load` function, and always call the matching
`*_release` function. A load operation validates that `model.algorithm` in the
package matches the object's static algorithm type and returns
`ANOM_STATUS_ALGORITHM_MISMATCH` when it does not.

```c
anom_patchcore_t patchcore = {0};
patchcore.struct_size = sizeof(patchcore);

anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
options.device = ANOM_DEVICE_GPU;
options.device_id = -1;

anom_status_t status = anom_patchcore_load(package_path, &options, &patchcore);
if (status == ANOM_STATUS_OK) {
    status = anom_patchcore_predict(&patchcore, &image, &prediction);
}
anom_prediction_release(&prediction);
anom_patchcore_release(&patchcore);
```

The algorithm structures intentionally contain only ABI bookkeeping and an
opaque implementation pointer. Functions stay as exported symbols instead of
being embedded function pointers, which allows the implementation to evolve
without mutating object layout.

Shared concepts remain shared: image and prediction descriptors, execution
options, model-package validation, package assembly, calibration, error codes,
and plugin loading. Algorithm capabilities do not share an interface. For
example, PatchCore and PaDiM each expose their own fitter type and options.
Future PatchCore memory-bank insertion, pruning, inspection, and persistence
operations belong only to `anom_patchcore_*`.

The implementation may reuse common runtime components internally. Internal
reuse does not create a public base object and is not part of the ABI contract.

## Legacy migration

The old generic session and fitter symbols remain implemented temporarily for
binary compatibility, but are hidden from the v2 public header by default.
Defining `ANOM_ENGINE_ENABLE_LEGACY_SESSION_API` exposes their declarations for
migration only. New code must use concrete algorithm objects.
