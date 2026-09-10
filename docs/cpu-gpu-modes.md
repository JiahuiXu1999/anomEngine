# CPU / GPU execution modes

The algorithm objects share the same C ABI in both modes. Device selection is a
load-time policy; the backend and provider are separate internal choices. GPU
means GPU neural-network inference and, for PatchCore/SPADE, GPU Faiss retrieval
unless permitted loading fallback selects CPU. Preprocessing, feature transforms
and postprocessing currently remain on CPU, and backend tensor outputs
are returned in host memory. ORT may also assign unsupported graph nodes to CPU.
Selecting CUDA is not a guarantee that every graph node executes on GPU.

## Selection contract

| Request | Candidate order |
| --- | --- |
| CPU | ONNX Runtime / CPU |
| GPU, no fallback | TensorRT / CUDA, ONNX Runtime / CUDA |
| GPU, load-only fallback | TensorRT / CUDA, ONNX Runtime / CUDA, ONNX Runtime / CPU |
| AUTO | TensorRT / CUDA, ONNX Runtime / CUDA, ONNX Runtime / CPU |

`backend_utf8` filters this list to the named backend. In particular, GPU plus
`"onnxruntime"` now requests ORT CUDA; it only permits CPU when load-only fallback
is enabled. AUTO permits CPU selection even with `ANOM_FALLBACK_NONE`, preserving
the existing API contract. `device_id = -1` selects device 0; an explicit CPU request
ignores a non-negative GPU device id. No CUDA provider is registered or probed on
the CPU load path.

Candidates must have compatible model artifacts. An engine-only package cannot
run on CPU: include ONNX to support CPU fallback. Existing engine load/build
policies still apply. Explicit FP16 currently restricts selection to TensorRT;
ORT does not automatically convert the model to FP16. FP16 is a TensorRT build
preference, and the precision field is not per-layer precision telemetry or proof
of the precision of a previously serialized engine.

Loading tries the next candidate only for retryable runtime failures. Invalid
manifests, corrupt artifacts and tensor-contract errors remain errors. A failed
warmup can select the next permitted candidate. After a successful load/warmup,
prediction errors are returned without changing backend or retrying on CPU.

`get_execution_info` reports the selected backend/provider, device id and the
previous failed attempts. A GPU-to-GPU candidate change is also recorded as a
fallback. The reported provider describes the neural-network runtime, not the
placement of the whole anomaly-detection pipeline.

## C usage

```c
anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
options.device = ANOM_DEVICE_GPU;
options.device_id = 0;
options.fallback = ANOM_FALLBACK_LOAD_ONLY;
options.precision = ANOM_PRECISION_AUTO;
/* Optional: options.backend_utf8 = "onnxruntime"; */

anom_direct_t direct = {0};
direct.struct_size = sizeof(direct);
if (anom_direct_init(&direct) != ANOM_STATUS_OK) return 1;
if (direct.load(&direct, model_path, &options) != ANOM_STATUS_OK) return 1;

anom_execution_info_t info = {0};
info.struct_size = sizeof(info);
if (direct.get_execution_info(&direct, &info) != ANOM_STATUS_OK) {
    direct.release(&direct);
    return 1;
}
/* Read info.backend_utf8, info.execution_provider_utf8, info.device_id,
   info.fallback_occurred and info.fallback_reason_utf8 before release. */
direct.release(&direct);
```

To request CPU, change `device` to `ANOM_DEVICE_CPU`. To require GPU inference,
use `ANOM_DEVICE_GPU` with `ANOM_FALLBACK_NONE`.

## Availability query

```c
anom_status_t status = anom_runtime_probe("onnxruntime", "cuda", 0, NULL);
/* OK: runtime/device available; UNSUPPORTED: provider/device unavailable.
   Plugin loading and invalid arguments have their own status codes.
   Use anom_get_last_error for the diagnostic. */
```

The query loads the backend plugin, checks provider availability and probes the
requested device. It does not open a model or build a TensorRT engine. CPU probing
does not initialize CUDA; CUDA probing may initialize its driver/provider. Query
success does not guarantee operator compatibility or enough memory for a model.
Always handle the result of `load`. CPU-only ORT distributions are valid even
when CUDA selection was enabled at build time: probing CUDA explains that the
provider is absent, and CPU execution remains available.

## Build and deployment

- `*-cpu` presets explicitly disable TensorRT, ORT CUDA and GPU Faiss. No CUDA Toolkit is
  required. Use a CPU ORT distribution for a minimal deployment.
- `*-nvidia` presets enable TensorRT, `ANOM_ENABLE_ORT_CUDA` and `ANOM_ENABLE_FAISS_GPU`. Install a GPU ORT
  distribution to use ORT CUDA, with the CUDA/cuDNN versions required by that
  distribution. Enabling this flag alone does not install a CUDA provider.
- ORT CUDA can be enabled independently of TensorRT:
  `-DANOM_ENABLE_TENSORRT=OFF -DANOM_ENABLE_ORT_CUDA=ON`.
- The ORT plugin calls CUDA provider APIs through ORT's C function table, and has
  no direct CUDA Toolkit link dependency. Its CPU path can still run without the
  optional CUDA provider DLLs. Windows deployment copies provider DLLs found next
  to the selected ORT runtime when CUDA support is enabled. Supply the provider's
  CUDA/cuDNN dependencies separately. On Linux, deploy ORT and its provider shared
  libraries together with their dependencies and configure the loader paths.

## Plugin compatibility and threading

Public ABI v3 structure sizes and existing field offsets are unchanged. Three
execution-info reserved words now report `faiss_provider`, `faiss_device_id` and
`faiss_fallback_occurred`. The original backend plugin v1 ABI is unchanged.
Backends optionally export `anom_backend_query_execution_v1`, which negotiates a
separate function table for explicit provider creation and device probing. A new
host still loads a legacy plugin in its original default mode. It rejects ORT CUDA
selection if the extension is missing, instead of allowing an old plugin to run
silently on CPU. A legacy plugin cannot answer the new probe API.

TensorRT load, inference and destruction select the object's owning CUDA device
and restore the calling thread's previous device. Pending inference transfers
are drained on error before host buffers are released. Backend inference remains
serialized per backend instance; callers must synchronize load/release against
in-flight object operations. Use separate algorithm objects for parallel workers.

## Validation

`c_api_contract_tests` covers CPU selection, strict GPU rejection, load-only
fallback, AUTO selection, diagnostics and runtime probing. Unavailable-device
tests use an invalid device id so results do not depend on CI GPU presence.
`ort_backend_tests` also loads a v1-only test plugin to verify legacy CPU loading
and rejection of CUDA requests when provider negotiation is unavailable.
`runtime_device_tests` compares CPU and GPU dynamic-batch results and invokes and
destroys a GPU backend on host threads different from its loading thread. CTest
reports hardware/provider-dependent tests as skipped (code 77) when unavailable;
once a runtime/device passes probing, load and inference failures fail the test.

GPU Faiss is provided by the optional `anom_search_faiss_cuda` plugin, with host
query/result buffers. See [Faiss CPU/GPU modes](faiss-cpu-gpu.md) for dependency,
fallback and verification details. Device-resident tensors and GPU preprocessing
remain separate extensions. They require an explicit device-memory and stream
contract across both algorithm and backend plugins; the current host tensor ABI
must not be used to pass device pointers.

## Host output ownership and copy cost

Backend plugin outputs are consumed as shared, read-only host-memory views.
The host no longer allocates and copies the complete output tensor batch when
crossing the backend plugin boundary. This applies to both TensorRT and ORT:
an output batch totaling N bytes avoids one N-byte host allocation/copy at that
boundary (N bytes read and N bytes written). This is not a claim about measured
end-to-end speedup, which depends on the model and the other pipeline stages.

Each retained Tensor shares ownership of the plugin batch, backend instance and
loaded DLL. The last owner releases the batch before destroying the instance or
unloading the DLL, including when the wrapper was already destroyed or the final
tensor is released on another thread. Later inference calls do not overwrite
retained outputs. Consumers read via const `Tensor::data()` and `byteSize()`;
`bytes` can be empty for an external view. Mutable `data()` access creates a
private copy so it cannot change another consumer's shared output.

TensorRT still performs device-to-host output transfers. ORT still copies its
runtime output into a host Tensor, and preprocessing, feature transforms
and postprocessing still run on CPU. GPU Faiss uploads the flattened CPU features
and returns distances/labels to CPU. The existing `backendMs` includes inference,
transfers and backend/plugin handling; compare it with `adapterMs` and `totalMs`
on production models before deciding the next GPU optimization.

`backend_memory_tests` verifies pointer identity on a 4 MiB feature output,
copy-on-write, retained results across inference calls, malformed/empty batch
cleanup, and cross-thread instance/DLL lifetime. It is part of the normal CTest
suite and can also run without OpenCV, ORT, Faiss or CUDA:

```sh
cmake -S tests/backend_memory -B build/backend-memory
cmake --build build/backend-memory --config Release
ctest --test-dir build/backend-memory -C Release --output-on-failure
```

The ORT provider setup follows the official
[CUDA provider configuration](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html),
including copies on the default stream. Consult its dependency table when choosing
an ORT CUDA distribution.
