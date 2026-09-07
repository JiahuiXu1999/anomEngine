# Algorithm object C ABI (v3)

Each algorithm is a caller-owned C structure containing its runtime state pointer
and a typed table of operations. Both C and C++ callers can create these objects
directly; no C++ wrapper is required.

| Algorithm | Public C header | C object |
|---|---|---|
| Direct Prediction | `anomEngine/direct.h` | `anom_direct_t` |
| EfficientAD | `anomEngine/efficientad.h` | `anom_efficientad_t` |
| DFKDE | `anomEngine/dfkde.h` | `anom_dfkde_t` |
| PaDiM | `anomEngine/padim.h` | `anom_padim_t` |
| PatchCore | `anomEngine/patchcore.h` | `anom_patchcore_t` |
| SPADE | `anomEngine/spade.h` | `anom_spade_t` |
| YOLO | `anomEngine/yolo.h` | `anom_yolo_t` |

Each header is independently usable and includes `anomEngine/common.h` for
shared types and model/result utilities. `anomEngine/anomEngine.h` remains the
umbrella header. PatchCore and PaDiM headers also define their algorithm-specific
fitter objects, options and operations. Structures are explicitly defined, so
algorithm-specific operations can be added without expanding every algorithm.

## Calling an object

1. Zero-initialize the whole structure and set `struct_size = sizeof(object)`.
2. Call the matching exported `*_init(&object)` and check its status.
3. Call operations through the structure, passing its address as `self`.
4. Release predictions with `anom_prediction_release` and the runtime with
   `object.release(&object)` before their storage goes out of scope.

```c
#include <anomEngine/patchcore.h>

anom_status_t detect(const char* package_path, const anom_image_t* image) {
    anom_patchcore_t patchcore = {0};
    patchcore.struct_size = sizeof(patchcore);
    anom_status_t status = anom_patchcore_init(&patchcore);
    if (status != ANOM_STATUS_OK) return status;

    anom_algorithm_options_t options = {0};
    options.struct_size = sizeof(options);
    options.device = ANOM_DEVICE_AUTO;
    options.device_id = -1;

    anom_prediction_t prediction = {0};
    prediction.struct_size = sizeof(prediction);
    status = patchcore.load(&patchcore, package_path, &options);
    if (status == ANOM_STATUS_OK) {
        status = patchcore.predict(&patchcore, image, &prediction);
        if (status == ANOM_STATUS_OK) {
            /* consume prediction.score, prediction.mask, ... */
        }
    }
    anom_prediction_release(&prediction);
    patchcore.release(&patchcore);
    return status;
}
```

The same code works in C++. Function pointers have no implicit `this`, so the
explicit `self` argument is required. All public functions and method pointers
use `ANOM_CALL` (cdecl on Windows). No C++ types, virtual tables or exceptions
are part of this boundary.

## Lifecycle and state

- Init fills every method and records `abi_version = ANOM_ENGINE_ABI_VERSION`.
  It does not load models or plugins. Init succeeds even for algorithms whose
  optional plugins are absent; load reports plugin availability failures.
- A missing/undersized object, incompatible nonzero ABI version or nonempty
  `internal` is rejected without modifying the object. Failed init does not
  guarantee callable methods. Error details use `anom_get_last_error`.
- A larger structure is accepted; init preserves the caller's `struct_size`
  and leaves bytes beyond the known structure untouched.
- Release clears runtime state and retains the methods, ABI version and size.
  Repeated release is safe on a valid object. The object can be loaded again.
  A failed load also leaves the method table available for retry.
- Do not copy an initialized object by assignment or memcpy. Do not modify
  `internal`, reserved fields or method pointers. Each object owns its state.
- Method pointers do not imply thread safety. Serialize operations on the same
  inference object. Fitter progress/cancellation retain their documented
  concurrency exceptions. Independent objects own independent runtimes.
- Keep the core DLL loaded while any object, method pointer or result is in use.
  Model-info strings are borrowed from the producing runtime/model and expire
  when it is released. C objects have no automatic destructor.

Loading checks the package's algorithm and returns
`ANOM_STATUS_ALGORITHM_MISMATCH` for a different algorithm.

## Fitting and future extensions

`anom_patchcore_fitter_init` and `anom_padim_fitter_init` populate the fitter
tables. Call `fitter.create(&fitter, &options)`, `add_batch`, `get_progress`,
`save_checkpoint`, `load_checkpoint`, `cancel`, `finalize`, and `release`
through the object. See [model lifecycle](cabi-model-lifecycle.md).

Future PatchCore memory-bank editing/pruning operations belong in its own
structure and implementation. They are not implemented by this interface
change. Do not insert/reorder existing fields when extending a published ABI;
negotiate new layouts and gate access by the caller's size/version.

## Migration and verification

ABI v3 enlarges algorithm and fitter objects. Rebuild clients with these headers
and deploy the matching core library; old v2 binaries are not supported as v3
clients. The former standalone algorithm and fitter operations are no longer
public symbols. Initialize the function table and call operations through the
object.

The former C++ wrappers have been removed. C and C++ use the same algorithm
headers and explicit object lifecycle. Migrate wrapper-based code to the C
object calls shown above. The pre-v2 generic session/fitter declarations remain
behind `ANOM_ENGINE_ENABLE_LEGACY_SESSION_API`.

The protocol checks can run without inference dependencies:

```sh
cmake -S tests/object_api -B build/cabi-v3-protocol
cmake --build build/cabi-v3-protocol
ctest --test-dir build/cabi-v3-protocol --output-on-failure
```

They link a C client against a shared library containing the production object
initializers and runtime test doubles. They check table dispatch, initialization,
size/version rejection, larger caller buffers, independent objects and reuse;
they do not validate numerical inference. The main `c_api_contract_tests` also
exercise real Direct inference and PatchCore fitting through object methods when
the required dependencies/plugins are available.
