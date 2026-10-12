# PatchCore 可选 CUDA 计算

`ANOM_ENABLE_CUDA_COMPUTE` 构建独立的 `anom_compute_cuda` 插件，默认 OFF，
`*-nvidia` 预设开启。CPU 主库和算法插件不链接 CUDA；CPU 模式不加载计算插件。
这与 `ANOM_ENABLE_FAISS_GPU` 是两个独立开关，计算插件不需要 GPU Faiss SDK。

## 覆盖范围与结构

| 阶段 | 实现 | 数据与精度 |
| --- | --- | --- |
| 预处理 | CUDA resize、中心裁剪、颜色转换、归一化、NCHW/NHWC 写入 | 支持灰度/BGR/BGRA、非连续图像；nearest/linear 在 GPU；其他插值保留 OpenCV resize，再在 GPU 处理其余步骤 |
| 特征聚合 | 平均池化；插值、层拼接、布局转换合并为一次写入 | 输出直接为 patch-major，避免额外 concat/flatten 缓冲；支持 mean aggregation |
| 原版 memory bank | cuBLAS 随机投影、CUDA 最小距离更新与 argmax | 原始高维向量形成最终 bank；同距离按较小下标选择；选中点不会再次入选 |
| MH/SubAD 拟合 | GPU 批均值/协方差、cuSOLVER 特征分解 | 保持双精度统计；分块合并仍使用原有 CPU 公式；MH merge/reduce 使用 CUDA coreset |
| MH 推理 | cuBLAS 投影、CUDA 白化缩放 | 双精度中间值，float32 输出 |
| SubAD 推理 | cuBLAS 投影/重建、CUDA 残差归约 | 双精度中间值；图像 top-fraction 小规模选择保留 CPU |
| 检索 | 复用可选 GPU Faiss 插件 | 新设备扩展支持显存查询和距离、主机 labels；不重复上传 memory bank |
| 评分 | 根据距离所在位置选择 CPU 或 CUDA | CPU 检索结果直接在 CPU 评分，避免为小归约上传；support 检索按 batch 合并 |
| 异常图 | CUDA bilinear resize、可分离 GaussianBlur | 与 CPU 相同的 Gaussian 权重与 REFLECT_101 边界 |

主机端封装位于 `src/compute/gpu_compute.*`，资源管理和 C ABI 分派位于
`src/compute/cuda_compute.cu`，设备算子位于 `src/compute/cuda_kernels.cuh`。
`src/plugin_api/anom_compute_plugin.h` 不暴露 C++、OpenCV 或 CUDA 对象。
8 位线性插值的逐阶段定点舍入与 [OpenCV resize.cpp](https://github.com/opencv/opencv/blob/4.x/modules/imgproc/src/resize.cpp) 对齐；测试在多种缩放比例、通道和布局下按 1e-6 浮点容差比较。

模型包校验、FAISS 文件保存、最终图像坐标恢复、归一化/阈值、形态学和区域分析继续使用原有 CPU 路径。

## 设备 Tensor 合约

`anom_device_plugin.h` 定义可协商的 backend/algorithm 扩展。
`device_id = -1` 表示主机缓冲，非负值表示对应 CUDA 设备。
原 v1 host-only Tensor ABI 的布局和含义不变；设备指针只能通过新扩展传递。

每次调用返回时，其输出已完成写入。当前版本使用明确的 stream 同步边界，不传递跨插件
stream/event 所有权。实例对内部 CUDA/cuBLAS/cuSOLVER 资源串行访问，并恢复调用线程原有设备。
设备 Tensor 保留分配者、插件 DLL 和必要的 backend instance；后续 infer 不覆盖已返回的输出，
最终引用可在其他线程释放。`Tensor::data()` 拒绝把设备指针当成主机地址。

支持扩展时的数据流为：

```text
主机图像 → CUDA 预处理 → TensorRT 显存输出 → CUDA 特征聚合
         → [MH 白化 / SubAD 残差] → [GPU Faiss] → 评分 / 异常图 → 主机最终结果
```

旧 backend 不支持设备输出时，会保留主机中转；CPU Faiss 或旧 GPU Faiss 插件也使用明确的
主机边界。`device_tensor_path` 只说明网络输出是否留在显存，不代表整个结果链路没有任何拷贝。

## 使用与回退

公共 `anom_algorithm_options_t` 和 `anom_patchcore_fitter_options_t` 增加 `compute_mode`，
使用一个原 reserved 字段，结构大小和已有字段偏移不变：

- `ANOM_COMPUTE_AUTO`（0）：网络在 GPU 时尝试计算插件，加载阶段不可用则使用 CPU 计算。
- `ANOM_COMPUTE_CPU`（1）：强制 CPU 计算；可以与 TensorRT 网络组合做旧链路性能基线。
- `ANOM_COMPUTE_CUDA`（2）：要求 CUDA 计算，缺少插件/设备时返回错误；需要 GPU 网络或 TensorRT 拟合模板。

```c
anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
options.device = ANOM_DEVICE_GPU;
options.fallback = ANOM_FALLBACK_LOAD_ONLY;
options.compute_mode = ANOM_COMPUTE_CUDA;
/* object.load(&object, package, &options); */
```

计算插件 AUTO 的可选回退与网络、Faiss 的加载策略独立。推理期间 CUDA 错误直接返回，
不会偷偷重跑 CPU。`get_execution_info` 新增 `compute_provider`（0 无该阶段、1 CPU、2 CUDA）
和 `device_tensor_path`，同样使用 reserved 字段。推理加载回退原因进入现有诊断字段。
拟合 AUTO 使用模板的后端设备；需要确认确实使用 GPU 的拟合任务应显式设 `ANOM_COMPUTE_CUDA`。

GPU 原版 coreset 始终执行投影空间的 greedy farthest-first，**不使用 CPU 的
`max_exact_distance_evaluations` 随机采样降级**。GPU 仍有 O(N*K*d) 运算量，显存或运行错误会返回。
做 CPU/GPU 采样性能对比时必须把 CPU 上限提高到覆盖 N*K，确保比较的是相同算法。
投影使用 FP32 GEMM，与 CPU 的 double 累加可能产生小误差；距离近乎相等时，选中点序列可以不同。
MH/SubAD 特征分解的特征向量符号/重根基底也可以不同，应比较残差、距离和最终分数。

## 构建、测试和基准

需要 CUDA Toolkit、兼容的宿主编译器，以及 cuBLAS/cuSOLVER。Windows 自动部署计算插件需要的
CUDA DLL，包括 cuSOLVER 的 cuSPARSE/nvJitLink 依赖。CPU-only 构建不需要这些依赖。

```sh
cmake -S . -B build/cuda -DANOM_ENABLE_CUDA_COMPUTE=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/cuda --config Release
ctest --test-dir build/cuda -C Release --output-on-failure
```

本次提交按要求不包含新增或修改的测试源码，也不包含新增 CUDA 测试目标的 CMake 注册。
以下测试工具和注册保留在本地工作区；干净检出只会运行仓库已提交的测试。

本地 `patchcore_cuda_tests` 将以下 12 项分别注册到 CTest：aggregate、projection、coreset、statistics、
preprocess、map、score、ownership、fitting、adapter、backend、session。
测试涵盖 CPU/GPU 数值、ties/重复向量、padding/stride、插值/裁剪/布局、非连续输入、非法输入、
PCA 重建、MH bank 距离、checkpoint、显存生命周期、C ABI 设备选择和 GPU fitter finalize。
无设备返回 CTest skip；缺少编译时承诺提供的插件是失败。CPU 回归测试独立运行。

```sh
# 整体或单项对照测试
build/cuda/bin/patchcore_cuda_tests --case coresetTest
# 2 次预热，7 次测量中位数；coreset 为 5 次
build/cuda/bin/patchcore_cuda_tests --benchmark
# 使用 deployment/model/wide_resnet50_2_patchcore.onnx；临时生成本机 TRT engine
# 224x224 输入、1024x1536 合成 bank；3 次预热，9 次测量中位数
build/cuda/bin/patchcore_cuda_tests --real-benchmark
```

Windows 多配置生成器的程序位于 `bin/Release`，带 `.exe` 后缀。
实际模型基准使用合成图像和 bank，验证计算一致性与耗时，不评估异常检测准确率。
报告 `compute_provider`、`faiss_provider`，避免将 CPU 检索混称为全 GPU 推理。
测试设备、原始数据、误差和实测收益见 [GPU 性能报告](patchcore-gpu-benchmark.md)。
