# anomEngine C ABI SDK 设计文档

> [!NOTE]
> 本文是 C ABI 改造前的历史设计草案，用于保留架构演进背景。当前实现已经采用运行时动态加载的算法与后端插件；涉及静态注册、目标名称和交付布局的内容均已被现有源码、根目录 README 和 `src/model/README.md` 取代。

> 状态：历史草案（v0.1，已被当前实现取代）
> 目标：把当前「单静态库 + C++ 头文件」的推理库，改造成「多 DLL + 纯 C 头文件」的可分发 SDK，按算法拆分 DLL，用户需要什么算法就链接什么。

---

## 1. 背景与目标

### 1.1 现状

- 构建产物是**一个静态库 `anom_model`**，算法、backend、预处理/后处理、基础设施全部编在一起。
- 对外接口是 **C++ 头文件**（`model/inference_session.h`），`cv::Mat`、`std::string`、`std::filesystem::path`、`Result<T>` 贯穿 API。
- 算法选择是**运行时**行为：manifest 里 `model.algorithm` 决定走哪个 adapter，`createAdapter()`（`src/adapters/adapter.cpp`）用 `switch` 静态 new 对应 adapter。
- 消费方必须 `#include` 源码头文件，且编译器/STL/OpenCV 版本必须与库完全一致。

### 1.2 目标

1. 交付物 = **一组 DLL + 一组纯 C 头文件**，不暴露任何 C++/OpenCV 类型。
2. **按算法拆分 DLL**：`anom_core.dll`（C ABI 唯一入口）+ 每个算法一个 `anom_adapter_<algo>.dll`。
3. **按需链接**：用户只链接 `anom_core.lib` + 他需要的 `anom_adapter_<algo>.lib`。
4. 语言无关：C/C++/C#/Python/Go 都能通过 C ABI 调用。

### 1.3 关键结论（来自代码分析）

- 推理链 **完全不依赖 `src/pipeline/`**（训练侧代码，仅被 `anom_model` 一并编译）。SDK 应将其剔除。
- adapter 只依赖 `model/` 头 + `adapters/feature_utils.{h,cpp}`，是天然「每算法一个 DLL」的边界。
- `feature_utils` 是共享张量工具（patchcore/padim/dfkde/efficientad/spade 复用），归入 `anom_core`。

---

## 2. 目标架构

```
最终用户应用（纯 C ABI）
   │  include anom_sdk.h + anom_adapter_<algo>.h
   │  link   anom_core.lib + anom_adapter_<algo>.lib
   ▼
anom_core.dll ────────────────────────────── C ABI 唯一入口
   ├─ C ABI 导出层（src/capi/）
   ├─ InferenceSession 生命周期（load/predict/destroy）
   ├─ 预处理 / 后处理（processing/）
   ├─ runtime backend（backends/: TensorRT + ONNX Runtime）
   ├─ 基础设施（infrastructure/: JSON + SHA256）+ feature_utils
   └─ adapter 注册表（按算法类型 → 工厂函数）
   ▲  anom_register_adapter()
anom_adapter_patchcore.dll / _padim / _direct / _efficientad
   / _dfkde / _spade / _yolo（每个一个 DLL，各导出 register 函数）
   ▲ 共享运行时
opencv_world.dll · TensorRT DLLs · faiss.dll · onnxruntime.dll · CUDA
```

---

## 3. C ABI 设计

### 3.1 命名与符号

- 所有对外符号统一 `anom_` 前缀。
- 导出宏：

```c
#if defined(_WIN32)
#  if defined(ANOM_BUILD_SHARED)
#    define ANOM_API __declspec(dllexport)
#  else
#    define ANOM_API __declspec(dllimport)
#  endif
#else
#  define ANOM_API __attribute__((visibility("default")))
#endif
```

### 3.2 错误码

镜像 `model/result.h` 的 `ErrorCode`，保持一对一，便于未来扩展与日志对照：

```c
typedef enum anom_status {
    ANOM_OK = 0,
    ANOM_ERR_INVALID_ARGUMENT,
    ANOM_ERR_INVALID_IMAGE,
    ANOM_ERR_IO,
    ANOM_ERR_INVALID_MANIFEST,
    ANOM_ERR_UNSUPPORTED_SCHEMA,
    ANOM_ERR_UNSUPPORTED_ALGORITHM,   /* 未链接/未注册对应算法 DLL */
    ANOM_ERR_ARTIFACT_MISSING,
    ANOM_ERR_ARTIFACT_PATH_ESCAPE,
    ANOM_ERR_ARTIFACT_CORRUPT,
    ANOM_ERR_TENSOR_SIGNATURE_MISMATCH,
    ANOM_ERR_TENSOR_SHAPE_MISMATCH,
    ANOM_ERR_TENSOR_TYPE_MISMATCH,
    ANOM_ERR_ENGINE_INCOMPATIBLE,
    ANOM_ERR_BACKEND_FAILURE,
    ANOM_ERR_ADAPTER_FAILURE,
    ANOM_ERR_CUDA_FAILURE,
    ANOM_ERR_OUT_OF_MEMORY,
    ANOM_ERR_NOT_INITIALIZED,
    ANOM_ERR_INTERNAL,
} anom_status_t;
```

### 3.3 全量头文件草案（`src/capi/anom_sdk.h`）

```c
#ifndef ANOM_SDK_H
#define ANOM_SDK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- 基础类型 ---- */
typedef enum anom_status anom_status_t;   /* 见 3.2 */
typedef struct anom_session    anom_session_t;    /* opaque */
typedef struct anom_prediction anom_prediction_t; /* opaque */

typedef enum anom_pixel_format {
    ANOM_FMT_BGR8 = 0,   /* 每像素 3 字节，B G R */
    ANOM_FMT_RGB8 = 1,   /* 每像素 3 字节，R G B */
    ANOM_FMT_GRAY8 = 2,  /* 每像素 1 字节 */
} anom_pixel_format_t;

typedef struct anom_region {
    int32_t x, y, width, height;  /* 外接框，原图坐标 */
    int32_t max_x, max_y;         /* 区域内最高分像素 */
    double  area;                 /* 像素面积 */
    float   mean_score;
    float   max_score;
} anom_region_t;

typedef struct anom_load_options {
    int backend;      /* -1 用 manifest 默认；0 TensorRT；1 ONNX Runtime */
    int ort_provider; /* -1 默认；0 CPU；1 CUDA */
    int engine_load_policy; /* -1 默认；0 engine_only；1 prefer_engine；2 build_if_missing */
    int fp16;         /* -1 默认；0/1 */
    int warmup;       /* 0/1 */
} anom_load_options_t;

/* ---- 会话生命周期 ---- */
ANOM_API anom_status_t anom_session_load(
    const char* package_path_utf8,
    const anom_load_options_t* options,  /* 可空 */
    anom_session_t** out_session);

ANOM_API void anom_session_destroy(anom_session_t* session);

/* ---- 推理 ---- */
ANOM_API anom_status_t anom_session_predict(
    anom_session_t* session,
    const void* pixels,          /* 首像素地址 */
    int32_t width,
    int32_t height,
    anom_pixel_format_t format,
    int32_t stride_bytes,        /* 每行字节数；<=0 表示紧凑排布 */
    anom_prediction_t** out_prediction);

/* ---- 结果读取（prediction 生命周期绑定，destroy 前有效） ---- */
ANOM_API float anom_prediction_score(const anom_prediction_t* p);        /* 归一化后分数 */
ANOM_API float anom_prediction_raw_score(const anom_prediction_t* p);    /* 原始分数 */
ANOM_API int   anom_prediction_is_anomalous(const anom_prediction_t* p);
ANOM_API const char* anom_prediction_model_id(const anom_prediction_t* p);
ANOM_API const char* anom_prediction_model_version(const anom_prediction_t* p);

/* anomaly map：float32 单通道，尺寸 w*h，归一化后的异常热力图；无空间输出时 data 为 NULL */
ANOM_API anom_status_t anom_prediction_get_anomaly_map(
    const anom_prediction_t* p, const float** data, int32_t* w, int32_t* h);
/* 二值 mask：uint8 单通道，0/255；未开启阈值化时 data 为 NULL */
ANOM_API anom_status_t anom_prediction_get_mask(
    const anom_prediction_t* p, const uint8_t** data, int32_t* w, int32_t* h);

/* 连通域分析结果（开启 postprocess.analyze 时有值） */
ANOM_API int anom_prediction_region_count(const anom_prediction_t* p);
ANOM_API anom_status_t anom_prediction_get_region(
    const anom_prediction_t* p, int32_t index, anom_region_t* out);
ANOM_API float anom_prediction_anomaly_area_ratio(const anom_prediction_t* p);

/* 耗时（毫秒） */
ANOM_API double anom_prediction_timing_preprocess_ms(const anom_prediction_t* p);
ANOM_API double anom_prediction_timing_backend_ms(const anom_prediction_t* p);
ANOM_API double anom_prediction_timing_adapter_ms(const anom_prediction_t* p);
ANOM_API double anom_prediction_timing_postprocess_ms(const anom_prediction_t* p);
ANOM_API double anom_prediction_timing_total_ms(const anom_prediction_t* p);

ANOM_API void anom_prediction_destroy(anom_prediction_t* prediction);

/* ---- 错误与版本 ---- */
ANOM_API const char* anom_last_error(void);              /* thread_local */
ANOM_API const char* anom_status_to_string(anom_status_t code);
ANOM_API const char* anom_version(void);

#ifdef __cplusplus
}
#endif
#endif /* ANOM_SDK_H */
```

### 3.4 C ABI 实现要点

- **句柄**：`anom_session_t` 内部持有 `std::unique_ptr<anom::model::InferenceSession>`；`destroy` 里 `delete`。分配与释放均在 core 内，跨 DLL 的 new/delete 配对安全。
- **图像零拷贝**：`cv::Mat(height, width, type, const_cast<void*>(pixels), stride)` 包装外部内存，预处理后即抛，不复制输入。
- **异常隔离**：每个 `extern "C"` 入口整体 `try { ... } catch (const std::exception& e) { setLastError(e.what()); return ...; } catch (...) { ... }`，异常绝不越出 C 边界。
- **UTF-8 路径**：Windows 下 `std::filesystem::path` 窄字符按 ACP 解释。入口先把 UTF-8 → `std::wstring` → `std::filesystem::path`，避免中文路径乱码。
- **错误信息**：`anom_last_error()` 用 `thread_local` 存储，天然线程安全。
- **结果内存模型**：`cv::Mat`（anomalyMap/mask）数据留在 `anom_prediction_t` 内部，`get_*_map` 返回裸指针，生命周期绑定到 `anom_prediction_destroy` 之前。

---

## 4. 算法按需接入（仅静态注册）

> 用户已确认：**仅静态注册**，不做 `LoadLibrary` 动态兜底。理由：依赖关系显式、链接期即可发现缺库、无 DLL 搜索路径的隐式约定。

### 4.1 注册表（core 内实现）

```cpp
// src/capi/adapter_registry.h —— 仅 SDK 内部使用
namespace anom::model {

using AdapterFactoryFn = IModelAdapter* (*)(const ModelPackage* package);

// core 导出；adapter DLL 的 register 函数调用它完成注册
extern "C" ANOM_API anom_status_t anom_register_adapter(
    AlgorithmType type, AdapterFactoryFn factory);

}  // namespace anom::model
```

- 注册表为进程内 `std::unordered_map<AlgorithmType, AdapterFactoryFn>` + `std::mutex`。
- `anom_register_adapter` 幂等：同类型重复注册返回 `ANOM_OK`（覆盖旧值），跨算法并发注册安全。

### 4.2 工厂函数：跨 DLL 的 C 桥

工厂用**裸函数指针**（C ABI 安全），参数/返回值是**不透明 C++ 指针**，两侧用 `reinterpret_cast` 还原。核心约束：**所有 DLL 由同一工具链（clang-cl）+ 同一 CRT（/MD）+ 共享 `opencv_world.dll` 构建**，见 §6。

```cpp
// anom_adapter_patchcore.dll 内部（patchcore_register.cpp）
#include "adapters/patchcore_adapter.h"
#include "capi/adapter_registry.h"

extern "C" ANOM_API anom_status_t anom_adapter_patchcore_register(void) {
    return anom::model::anom_register_adapter(
        anom::model::AlgorithmType::PatchCore,
        [](const anom::model::ModelPackage* pkg) -> anom::model::IModelAdapter* {
            const auto& manifest = pkg->manifest();
            const auto& cfg = std::get<anom::model::PatchCoreConfig>(manifest.algorithmConfig);
            return new anom::model::PatchCoreAdapter(cfg, manifest.outputs);
        });
}
```

`createAdapter()` 改造为：

```cpp
Result<std::unique_ptr<IModelAdapter>> createAdapter(const ModelPackage& package) {
    auto factory = adapterRegistry().find(package.manifest().algorithm);
    if (!factory) {
        return Status::error(ErrorCode::UnsupportedAlgorithm,
            "No adapter registered for this algorithm; link the matching anom_adapter_<algo> library");
    }
    std::unique_ptr<IModelAdapter> adapter(factory(&package));
    auto loaded = adapter->loadAssets(package);
    if (!loaded) return loaded.status();
    return adapter;
}
```

### 4.3 用户接入方式

**方式 A：C++ 用户（链接即自动注册）**

每个算法提供一个头文件，用静态初始化器在进程启动时自动注册：

```cpp
// include/anom_adapter_patchcore.h
#pragma once
#include "anom_sdk.h"
extern "C" ANOM_API anom_status_t anom_adapter_patchcore_register(void);

#if defined(_MSC_VER)
#  pragma comment(lib, "anom_core.lib")
#  pragma comment(lib, "anom_adapter_patchcore.lib")
#endif

// 自动注册：链接本库即生效，无需手动调用 register
namespace anom_detail {
    inline static const bool _patchcore_auto =
        (anom_adapter_patchcore_register(), true);
}
```

**方式 B：纯 C 用户（显式注册）**

```c
#include "anom_sdk.h"
extern anom_status_t anom_adapter_patchcore_register(void);

int main(void) {
    anom_adapter_patchcore_register();   /* 在首次 anom_session_load 之前调用 */
    anom_session_t* s = NULL;
    anom_session_load("model_pkg", NULL, &s);
    ...
}
```

> 约定：无论哪种方式，`register` 都必须在首次 `anom_session_load` 之前完成；重复调用幂等。

### 4.4 未注册行为

`anom_session_load` 遇到未注册算法 → 返回 `ANOM_ERR_UNSUPPORTED_ALGORITHM`，`anom_last_error()` 给出「请链接 anom_adapter_<algo>」的提示。这正是「需要什么算法就链接什么」的边界。

---

## 5. CMake 构建改造

### 5.1 目标清单

| 目标 | 类型 | 内容 |
|---|---|---|
| `anom_core` | SHARED | model/ + processing/ + backends/ + infrastructure/ + adapters/feature_utils.cpp + capi/ |
| `anom_adapter_patchcore` 等 ×7 | SHARED | 各 `xxx_adapter.cpp` + `xxx_register.cpp` |
| ~~pipeline~~ | 移除 | 训练侧，不进 SDK |
| `model_contract_tests` / `ort_backend_tests` | exe | 保持，链接 `anom_core` |

### 5.2 关键点

```cmake
# core 定义 ANOM_BUILD_SHARED 以 dllexport；adapter/测试消费时 dllimport
target_compile_definitions(anom_core PRIVATE ANOM_BUILD_SHARED=1)
# 对 core 打开 Windows 全符号导出（C++ 内部符号需被 adapter 引用）
set_target_properties(anom_core PROPERTIES WINDOWS_EXPORT_ALL_SYMBOLS ON)
# 统一动态 CRT，保证跨 DLL 的 STL 对象安全
if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")
endif()

# 每个 adapter：
add_library(anom_adapter_patchcore SHARED
    src/adapters/patchcore_adapter.cpp
    src/capi/patchcore_register.cpp)
target_link_libraries(anom_adapter_patchcore PRIVATE anom_core)

# 安装
install(TARGETS anom_core ${ANOM_ADAPTER_TARGETS}
    RUNTIME DESTINATION bin   LIBRARY DESTINATION lib   ARCHIVE DESTINATION lib)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/src/capi/anom_sdk.h
              ${ANOM_ADAPTER_HEADERS} DESTINATION include)
```

- 复用现有 `anom_deploy_runtime_dependencies`，把 OpenCV/TensorRT/FAISS/ORT 运行时 DLL 部署到安装 `bin/`。
- `feature_utils.cpp` 从「每个 adapter 各编一份」改为「core 编译并导出符号」，避免重复符号。

---

## 6. 关键风险与约束

| # | 风险 | 应对 |
|---|---|---|
| 1 | **跨 DLL 的 C++ ABI**：`IModelAdapter*`、`TensorMap`、`cv::Mat`、虚函数、STL 对象跨 core↔adapter 边界传递 | 强制「同一工具链（clang-cl）+ 统一 `/MD` 动态 CRT + 共享 `opencv_world.dll`」。最终用户只走纯 C ABI，不受此约束。当前构建已满足 |
| 2 | **异常越界** | 每个 C 入口 try/catch(...) 转错误码 |
| 3 | **中文路径乱码** | UTF-8 → `std::wstring` → `std::filesystem::path` |
| 4 | **符号导出** | `ANOM_API` 宏 + `WINDOWS_EXPORT_ALL_SYMBOLS`（core 内部符号供 adapter 引用） |
| 5 | **内存所有权** | 谁分配谁释放；`session`/`prediction` 的 new/delete 都收口在 core |
| 6 | **注册时序** | 文档强制：register 先于首次 load；幂等 |
| 7 | **静态注册的取舍** | 换算法需重新链接，但换来链接期可校验 + 无 DLL 搜索路径隐式约定（用户已确认接受） |

---

## 7. 分阶段实施计划

| 阶段 | 内容 | 验收标准 |
|---|---|---|
| P1 | 抽 `anom_core` 共享库，摘除 pipeline，现有测试切到链接 core | `ctest` 全绿 |
| P2 | 新增 `src/capi/`：C ABI 层 + `anom_sdk.h`，包 `InferenceSession` | 纯 C 客户端能 load/predict/destroy |
| P3 | 拆 7 个 adapter DLL + 注册表 + 工厂 C 桥 | 只链 patchcore 时，其他算法返回 `UNSUPPORTED_ALGORITHM` |
| P4 | 符号宏、install 规则、运行时依赖部署 | `cmake --install` 产出完整 bin/include/lib |
| P5 | `examples/c_client.c` + 用户接入文档 | 按文档可零障碍接入 |

---

## 8. 待确认事项

1. DLL 命名：`anom_core.dll` / `anom_adapter_patchcore.dll` 是否符合预期？（可改为 `libanom_core.dll` 等）
2. 是否需要同时提供 `predict_batch` 的 C ABI（当前 C++ 有 `predictBatch`）？
3. `anom_load_options_t` 覆盖的 override 字段是否够用（目前仅 backend/provider/load_policy/fp16/warmup）？
4. 是否需要导出 `model_info`（signature、graph_contract 等）的只读查询接口？
