# anomEngine

[English](README.md) | **简体中文**

跨平台 C++20 视觉异常检测推理 SDK，提供稳定的 C ABI、可插拔算法，以及 TensorRT / ONNX Runtime 后端。

> [!IMPORTANT]
> anomEngine 目前处于早期开发阶段。在首个稳定版本发布之前，API、模型清单（manifest）和部署行为都可能发生变化。
> 未经独立验证，暂不建议将其用于安全关键场景或生产环境。
> Windows 构建已使用 MSVC 验证；目前已提供 Linux 构建预设和源码支持，Linux 发布验证仍在进行中。

## 主要特性

- 带版本管理的 C11 ABI，确保 STL、OpenCV 类型、异常和 C++ 虚接口不跨越公共 DLL / 共享库边界。
- 每种算法都是独立的公共对象（`anom_patchcore_t`、`anom_padim_t`、`anom_yolo_t` 等），算法专属能力无需扩充统一会话接口即可独立演进。
- 算法插件目前支持 Direct Prediction、EfficientAD、DFKDE、PaDiM、PatchCore、SPADE，以及 YOLO 风格的预测输出。
- 后端插件支持 NVIDIA GPU 上的 TensorRT，以及 CPU / 可选 CUDA 上的 ONNX Runtime。
- 通过模型包清单声明张量绑定、预处理、后处理、运行时设置，以及可选的模型资源 SHA-256 校验和。
- 共享后处理模块通过统一的结果接口提供图像分数、异常图、掩码、连通区域和耗时信息。
- 提供适用于 Windows MSVC 和 Linux GCC / Clang 的 CMake 预设。

## 架构

应用程序只需链接 `anomEngine` 核心库，并直接创建带函数指针成员的 C ABI 算法结构体。
按需包含独立头文件，例如 [`patchcore.h`](include/anomEngine/patchcore.h)，
或通过 [`anomEngine.h`](include/anomEngine/anomEngine.h) 包含全部算法。
设置 `struct_size` 后，调用 `anom_patchcore_init(&object)` 初始化函数表，
再通过 `object.load(&object, path, &options)`、
`object.predict(&object, &image, &prediction)` 完成检测链路。
公共 ABI 已升至 v3，旧 v2 客户端需要重新编译并部署配套核心库。
C 和 C++ 用户统一直接使用这些 C ABI 结构体。
运行时，每个对象会校验模型包的算法类型，再通过支持版本协商的 C 函数表加载对应算法插件和后端插件。

```text
应用程序
    -> Direct | EfficientAD | DFKDE | PaDiM | PatchCore | SPADE | Yolo
        -> anomEngine 共享运行时
            -> 对应的算法插件
            -> 后端插件：TensorRT | ONNX Runtime
```

有关 SDK 接口约定、模型包格式、运行时行为和完整的 C API 示例，请参阅
[`src/model/README.md`](src/model/README.md)。正式的清单 schema 和示例位于 [`src/model`](src/model)。
具体算法对象 API 与旧接口迁移策略见
[`docs/algorithm-object-api.md`](docs/algorithm-object-api.md)。
模型验证、原子化模型包组装、能力发现，以及 PatchCore / PaDiM 拟合 API 的说明见
[`docs/cabi-model-lifecycle.md`](docs/cabi-model-lifecycle.md)。

## 环境要求

- CMake 3.25 或更高版本
- C11 编译器和 C++20 编译器
- OpenCV（`core` 和 `imgproc` 模块）
- 构建 PatchCore、SPADE 或完整测试套件时，需要 FAISS
- 当 `ANOM_ENABLE_TENSORRT=ON` 时，需要 TensorRT 和 CUDA Toolkit
- 当 `ANOM_ENABLE_ONNXRUNTIME=ON` 时，需要 ONNX Runtime

本仓库不分发第三方 SDK 和数据集。请使用系统中已安装的依赖，或通过下列 CMake 缓存变量指定 SDK 的绝对路径，并遵守各依赖项的许可条款。

| 变量 | 用途 |
|---|---|
| `ANOM_OPENCV_ROOT` | 包含 `OpenCVConfig.cmake` 的目录 |
| `ANOM_FAISS_ROOT` | FAISS C/C++ 发行包根目录 |
| `ANOM_TENSORRT_ROOT` | TensorRT 发行包根目录 |
| `ANOM_ONNXRUNTIME_ROOT` | ONNX Runtime C/C++ 发行包根目录 |

## 构建

原有预设会启用全部算法、两个后端、严格警告和测试。此外还提供明确的 `-cpu` 和 `-nvidia` 变体：CPU 预设不需要 TensorRT 或 CUDA；NVIDIA 预设启用 TensorRT，同时启用可选的 ONNX Runtime CUDA 支持，并保留 CPU 回退。配置好所需依赖的查找路径后，运行与平台和部署方式匹配的预设：

```powershell
cmake --preset windows-msvc-ninja-release
cmake --build --preset windows-msvc-ninja-release
ctest --preset windows-msvc-ninja-release
```

例如，Windows 上仅使用 CPU 的构建方式如下：

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

如果只需要 ONNX Runtime 后端，不需要依赖 FAISS 的插件和测试，可以使用以下精简构建配置：

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

在 Windows 上，CMake 路径请使用正斜杠，例如 `D:/sdk/onnxruntime`。

## 仓库结构

CPU/GPU 选择、运行时探测、回退规则和 ORT CUDA 部署说明见
[`docs/cpu-gpu-modes.md`](docs/cpu-gpu-modes.md)。GPU 模式当前加速神经网络推理，
预处理、Faiss 检索和后处理仍在 CPU 上执行。

| 路径 | 职责 |
|---|---|
| `include/anomEngine` | 公共 C API |
| `src/adapters` | 算法适配器 |
| `src/backends` | TensorRT 和 ONNX Runtime 实现 |
| `src/plugins` | 动态插件入口和加载逻辑 |
| `src/processing` | 预处理、后处理和分析 |
| `src/model` | 会话、清单、结果类型、schema 和示例 |
| `src/pipeline` | 训练侧 / 统计数据准备辅助功能 |
| `tests` | ABI、模型接口约定和后端测试 |
| `docs` | 设计说明 |
| `tools` | 模型导出工具 |

## 安全与模型包

模型包在加载边界处被视为不可信输入。模型资源路径必须位于模型包根目录内，清单可以使用小写 SHA-256 校验和锁定资源内容。对于来自不可信来源的模型或插件，应先审查模型包内容并验证完整的部署环境，再进行部署。

## 参与贡献

项目仍在确立公共接口。在投入较大改动之前，请先创建 issue，说明使用场景和提议的接口约定。请保持改动范围集中，维护 C ABI 边界，为可观察行为添加测试，并在提交 pull request 前运行相关 CMake / CTest 预设。

## 许可证

anomEngine 采用 [Apache License 2.0](LICENSE) 许可证。第三方库、模型文件和数据集仍受各自条款约束。
