# anomEngine

**English** | [简体中文](README.zh-CN.md)

Cross-platform C++20 visual anomaly-detection inference SDK with a stable C ABI,
pluggable algorithms, and TensorRT/ONNX Runtime backends.

> [!IMPORTANT]
> anomEngine is in early development. APIs, manifests, and deployment behavior
> may change before the first stable release. It is not yet recommended for
> safety-critical or production use without independent validation.
> Windows builds are validated with MSVC; Linux presets and source support are
> available, while Linux release validation is still in progress.

## Highlights

- A versioned C11 ABI keeps STL, OpenCV types, exceptions, and C++ virtual
  interfaces out of the public DLL/shared-library boundary.
- Algorithm plugins currently cover Direct Prediction, EfficientAD, DFKDE,
  PaDiM, PatchCore, SPADE, and YOLO-style prediction outputs.
- Backend plugins support TensorRT on NVIDIA GPUs and ONNX Runtime on CPUs.
- A manifest-driven model package declares tensor bindings, preprocessing,
  postprocessing, runtime settings, and optional SHA-256 artifact checksums.
- Shared postprocessing produces image scores, anomaly maps, masks, connected
  regions, and timing information through one consistent result contract.
- CMake presets are provided for MSVC on Windows and GCC/Clang on Linux.

## Architecture

Applications link only the `anomEngine` core library and include
[`include/anomEngine/anomEngine.h`](include/anomEngine/anomEngine.h). At runtime,
the core reads a model package manifest and loads the requested algorithm and
backend plugins through version-negotiated C function tables.

```text
application
    -> anomEngine C ABI
        -> algorithm plugin: direct | efficientad | dfkde | padim | patchcore | spade | yolo
        -> backend plugin:   TensorRT | ONNX Runtime
```

See [`src/model/README.md`](src/model/README.md) for the SDK contract, model
package format, runtime behavior, and a complete C API example. The formal
manifest schema and examples live in [`src/model`](src/model).
The model validation, atomic package assembly, capability discovery, and
PatchCore/PaDiM fitting APIs are documented in
[`docs/cabi-model-lifecycle.md`](docs/cabi-model-lifecycle.md).

## Requirements

- CMake 3.25 or newer
- A C11 compiler and a C++20 compiler
- OpenCV (`core` and `imgproc`)
- FAISS when building PatchCore, SPADE, or the full test suite
- TensorRT and the CUDA Toolkit when `ANOM_ENABLE_TENSORRT=ON`
- ONNX Runtime when `ANOM_ENABLE_ONNXRUNTIME=ON`

Third-party SDKs and datasets are not distributed in this repository. Supply
system installations or absolute SDK roots through the CMake cache variables
listed below, and follow each dependency's own license terms.

| Variable | Purpose |
|---|---|
| `ANOM_OPENCV_ROOT` | Directory containing `OpenCVConfig.cmake` |
| `ANOM_FAISS_ROOT` | FAISS C/C++ distribution root |
| `ANOM_TENSORRT_ROOT` | TensorRT distribution root |
| `ANOM_ONNXRUNTIME_ROOT` | ONNX Runtime C/C++ distribution root |

## Build

The original presets enable all algorithms, both backends, strict warnings, and
tests. Explicit `-cpu` and `-nvidia` variants are also available; CPU presets do
not require TensorRT or CUDA, while NVIDIA presets enable TensorRT alongside the
CPU-only ONNX Runtime fallback. After making the required dependencies discoverable, run
the preset matching your platform and deployment:

```powershell
cmake --preset windows-msvc-ninja-release
cmake --build --preset windows-msvc-ninja-release
ctest --preset windows-msvc-ninja-release
```

For example, a CPU-only Windows build uses:

```powershell
cmake --preset windows-msvc-ninja-release-cpu
cmake --build --preset windows-msvc-ninja-release-cpu
ctest --preset windows-msvc-ninja-release-cpu
```

```bash
cmake --preset linux-gcc-ninja-release
cmake --build --preset linux-gcc-ninja-release
ctest --preset linux-gcc-ninja-release
```

For a smaller ONNX Runtime-only build without the FAISS-based plugins or tests:

```bash
cmake -S . -B build/local \
  -DANOM_OPENCV_ROOT=/absolute/path/to/opencv \
  -DANOM_ONNXRUNTIME_ROOT=/absolute/path/to/onnxruntime \
  -DANOM_ENABLE_TENSORRT=OFF \
  -DANOM_BUILD_ALGO_PATCHCORE=OFF \
  -DANOM_BUILD_ALGO_SPADE=OFF \
  -DBUILD_TESTING=OFF
cmake --build build/local --config Release
```

On Windows, use forward slashes in CMake paths, for example
`D:/sdk/onnxruntime`.

## Repository layout

| Path | Responsibility |
|---|---|
| `include/anomEngine` | Public C API |
| `src/adapters` | Algorithm adapters |
| `src/backends` | TensorRT and ONNX Runtime implementations |
| `src/plugins` | Dynamic plugin entry points and loading |
| `src/processing` | Preprocessing, postprocessing, and analysis |
| `src/model` | Sessions, manifests, result types, schema, and examples |
| `src/pipeline` | Training-side/statistics preparation helpers |
| `tests` | ABI, model-contract, and backend tests |
| `docs` | Design notes |
| `tools` | Model export utilities |

## Security and model packages

Model packages are treated as untrusted input at their load boundary. Artifact
paths must remain inside the package root, and manifests may pin artifacts with
lowercase SHA-256 checksums. Do not deploy models or plugins from untrusted
sources without reviewing the package contents and validating the complete
deployment environment.

## Contributing

The project is still establishing its public interfaces. Before investing in a
large change, open an issue describing the use case and proposed contract. Keep
changes focused, preserve the C ABI boundary, add tests for observable behavior,
and run the relevant CMake/CTest preset before submitting a pull request.

## License

anomEngine is licensed under the [Apache License 2.0](LICENSE). Third-party
libraries, model files, and datasets remain subject to their respective terms.
