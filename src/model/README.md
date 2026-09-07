# anomEngine inference SDK

The supported consumer boundary is the versioned C ABI in `include/anomEngine/anomEngine.h`. Applications link
only `anomEngine.lib` / `anomEngine.dll`; the core selects and dynamically loads the required algorithm and
backend plugins from the model manifest. C++ implementation headers under `src` are private and are not part of
the SDK ABI.

## SDK layout

```text
include/anomEngine/anomEngine.h
include/anomEngine/common.h
include/anomEngine/patchcore.h (and one .h per algorithm)
lib/anomEngine.lib
bin/anomEngine.dll
bin/anom_algo_<algorithm>.dll
bin/anom_backend_<backend>.dll
```

By default plugins are resolved next to `anomEngine.dll`. `anom_algorithm_options_t::plugin_directory_utf8` can
select a different explicit directory. Plugins are loaded with a fixed, version-negotiated C function table;
STL, OpenCV types, exceptions, and C++ virtual interfaces never cross a DLL boundary. Memory returned by the
public API is released with `anom_prediction_release`.

## Source layout

| Directory | Responsibility |
|---|---|
| `src/model` | Shared runtime implementation, manifest/package contract, status and tensor/result types |
| `src/backends` | Runtime abstraction, factory, TensorRT and ONNX Runtime implementations |
| `src/adapters` | PatchCore, PaDiM and direct-prediction algorithm adapters |
| `src/pipeline` | Training-side PatchCore memory-bank and PaDiM statistics pipelines |
| `src/processing` | Image preprocessing and anomaly-result postprocessing |
| `src/infrastructure` | Internal JSON parser and SHA-256 artifact verification |

Internal C++ tests may include `model/inference_session.h`, but SDK consumers
should include an algorithm header such as `anomEngine/patchcore.h`, or the
`anomEngine/anomEngine.h` umbrella. See `docs/algorithm-object-api.md` for ABI v3
object initialization and migration requirements.

## Algorithm-specific entry points

The public API has no generic inference session. A caller chooses a concrete
algorithm object, and loading fails if the model package declares a different
algorithm. Shared image, prediction, execution, and error types remain common.

```c
#include <anomEngine/anomEngine.h>

anom_padim_t padim = {0};
padim.struct_size = sizeof(padim);
if (anom_padim_init(&padim) != ANOM_STATUS_OK) return 1;

anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
if (padim.load(&padim, "models/bottle-padim/1.0.0", &options) != ANOM_STATUS_OK) {
    return 1;
}

anom_image_t image = {0};
image.struct_size = sizeof(image);
image.data = pixels;
image.width = width;
image.height = height;
image.stride_bytes = stride;
image.pixel_format = ANOM_PIXEL_FORMAT_BGR8;

anom_prediction_t prediction = {0};
prediction.struct_size = sizeof(prediction);
if (padim.predict(&padim, &image, &prediction) == ANOM_STATUS_OK) {
    /* consume prediction.score, prediction.mask, prediction.regions, ... */
    anom_prediction_release(&prediction);
}
padim.release(&padim);
```

C++ applications use the same C structures, member calls and explicit release
shown above. No separate wrapper header is needed.

### CPU/GPU selection

Every algorithm accepts the same execution options without sharing an
algorithm interface:

```c
anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
options.device = ANOM_DEVICE_GPU;
options.device_id = -1;
options.fallback = ANOM_FALLBACK_LOAD_ONLY;
options.precision = ANOM_PRECISION_AUTO;

anom_patchcore_t patchcore = {0};
patchcore.struct_size = sizeof(patchcore);
if (anom_patchcore_init(&patchcore) != ANOM_STATUS_OK ||
    patchcore.load(&patchcore, model_path, &options) != ANOM_STATUS_OK) {
    return 1;
}

anom_execution_info_t execution = {0};
execution.struct_size = sizeof(execution);
patchcore.get_execution_info(&patchcore, &execution);
/* execution.backend_utf8, execution.execution_provider_utf8 and
   execution.fallback_reason_utf8 describe the actual runtime. */
patchcore.release(&patchcore);
```

`ANOM_DEVICE_CPU` selects ONNX Runtime CPU. `ANOM_DEVICE_GPU` selects TensorRT;
`ANOM_FALLBACK_LOAD_ONLY` additionally permits ONNX Runtime CPU when GPU initialization fails. `ANOM_DEVICE_AUTO`
tries TensorRT/CUDA and then ONNX Runtime/CPU. A non-empty `backend_utf8` constrains
selection to `tensorrt` or `onnxruntime`. Fallback only occurs while loading or warming the algorithm object; prediction
failures are never silently retried on a different device.

The package root must contain `manifest.json`. All artifact paths are relative to this root and are rejected if
they are absolute or escape through `..`. An optional `checksums` object maps artifact paths to lowercase SHA-256
digests; every listed artifact is verified before any model or index is loaded. Generated engine caches normally
remain outside this immutable checksum set.

## Graph contracts

- `feature_pyramid`: the selected backend returns named NCHW feature maps. `PatchCoreAdapter` or `PadimAdapter` computes the
  anomaly prediction using immutable algorithm assets.
- `prediction`: the selected backend returns an image score and, optionally, an anomaly map. `DirectPredictionAdapter` only
  converts those tensors to the standard result.

Tensor names are never guessed. The `outputs` object binds stable semantic names to exact graph tensor names.

## Postprocessing & analysis contract

Every algorithm adapter converges on a single intermediate — `RawPrediction` — and never performs
postprocessing itself. `AnomalyPostprocessor` then runs one shared, configurable stage chain in a fixed order:

```
geometry restore -> normalize -> smooth -> threshold -> morphology -> analyze
```

- The adapter emits `score` in its raw scale and `anomalyMap` as a floating-point `CV_32FC1` score map in
  pre-processed coordinates at native resolution. Higher always means more anomalous. An empty `anomalyMap` is
  legal (models without a spatial head) and degrades the chain to image-level-only processing.
- `normalize` rescales scores to `[0, 1]` using the `*_min` / `*_threshold` / `*_max` bounds; when disabled the
  raw scale is kept and thresholds are compared directly.
- `smooth` applies a Gaussian blur to the map before thresholding (`smooth_kernel`, `smooth_sigma`).
- `threshold` produces the binarized `mask` from `pixel_threshold` / `pixel_sensitivity`.
- `morphology` cleans the mask with an `open` or `close` operation.
- `analyze` runs connected-component analysis (`AnomalyAnalyzer`) and populates `AnomalyAnalysis`: per-region
  bounding boxes, areas, and score statistics (sorted by descending area), plus image-level aggregates
  (`anomalyAreaRatio`, `meanScore`, `maxScore`, `regionCount`). `min_region_area` and `max_regions` control
  filtering and truncation.

All stages are optional and default to their historical no-op behavior, so existing manifests keep working.

## PaDiM statistics format

`statistics` is a little-endian binary file:

| Field | Type | Count |
|---|---:|---:|
| magic (`ANPADIM\0`) | byte | 8 |
| format version (`1`) | uint32 | 1 |
| feature height | uint32 | 1 |
| feature width | uint32 | 1 |
| selected dimension | uint32 | 1 |
| mean, layout `L,D` | float32 | `H*W*D` |
| precision, layout `L,D,D` | float32 | `H*W*D*D` |

`channel_indices` is exactly `D` little-endian `int32` values. Indices must be unique and non-negative. The
runtime never generates indices or fits Gaussian statistics.

## Runtime behavior

- `runtime.backend` selects `tensorrt` (the backwards-compatible default) or `onnxruntime`.
- ONNX Runtime uses the CPU execution provider. NVIDIA GPU execution is handled exclusively by TensorRT.
- ORT discovers every graph input/output from model metadata, preserves dynamic dimensions in the signature,
  validates concrete request shapes and copies resolved dynamic outputs into the backend-neutral `TensorMap`.
- ORT thread counts, sequential/parallel execution, graph optimization, memory pattern, CPU arena, and
  JSON profiling are controlled by the manifest. Zero thread counts delegate sizing to ORT.
- `fp16`, `workspace_mb`, engine cache, and engine load policy remain TensorRT settings. ORT preserves the data
  types exported in the ONNX graph and does not silently rewrite an FP32 model to FP16.
- TensorRT engine loading and ONNX building are controlled by `engine_only`, `prefer_engine`, or
  `build_if_missing`.
- Dynamic output shapes are resolved from the execution context before allocation.
- Device buffers grow on demand and are reused. A non-blocking CUDA stream performs asynchronous transfers.
- A backend instance serializes access to its execution context. Multiple sessions can be used for concurrent
  execution until a context-pool implementation is introduced.
- Every load boundary validates data type, rank, semantic output bindings, algorithm asset dimensions, and file
  lengths before inference.

See `manifest.schema.json`, `examples/patchcore.manifest.json`, `examples/padim.manifest.json`, and
`examples/padim-ort.manifest.json` for the formal schema and complete examples.

## Building with ONNX Runtime

The checked-in presets build and test the complete SDK:

```powershell
cmake --preset windows-msvc-ninja-release
cmake --build --preset windows-msvc-ninja-release
ctest --preset windows-msvc-ninja-release
```

The `RelWithDebInfo` presets emit debugging information while retaining a Release-compatible CRT. On Windows,
avoid a true `/MDd` Debug build when using prebuilt FAISS or ORT packages that were compiled against the Release
CRT, because mixing those runtimes is not ABI-safe.

Place an official C/C++ ONNX Runtime distribution under `libs/onnxruntime`, or point CMake at one explicitly:

```powershell
cmake -S . -B build -DANOM_ONNXRUNTIME_ROOT=D:/sdk/onnxruntime
```

The backend is enabled by default and configuration fails early when its headers, import library, or Windows DLL
are missing. Use `-DANOM_ENABLE_ONNXRUNTIME=OFF` for a TensorRT-only build. Only the CPU execution provider is
supported; NVIDIA GPU execution is provided by the TensorRT backend.

`examples/padim-ort.manifest.json` shows a complete PaDiM package configuration. PaDiM itself remains backend
independent: its adapter consumes the same explicitly named feature tensors from ORT or TensorRT.
