# PatchCore GPU 性能与验证报告

测量日期：2026-10-12。以下是本机实测，不代表其他设备、输入规模或 memory bank 的固定收益。
实现架构、开关与复现入口见 [PatchCore 可选 CUDA 计算](patchcore-gpu.md)。

## 环境和方法

- Windows，Intel Core i5-13400F（10 核、16 逻辑处理器），NVIDIA RTX 5070 Ti 16 GB。
- NVIDIA 驱动 616.64，CUDA Toolkit 13.2.78，计算架构 120。
- MSVC 19.51.36257，Release / Ninja，OpenCV 4.13，TensorRT 10.16.1.11。
- GPU 构建启用 TensorRT、ONNX Runtime、CUDA compute；`ANOM_ENABLE_FAISS_GPU=OFF`。
- 各阶段分别测量 CPU 和 GPU，2 次预热后取 7 次耗时中位数，coreset 为 5 次；不与其他测试并行运行。
- 计时包含同步完成，不把 CUDA 异步提交时间当成计算时间；默认库线程配置，未锁定 GPU 频率或 CPU 亲和性。
- CPU 基线为仓库现有算法，未另外为 CPU 实现 SIMD/GEMM 优化。随机投影 CPU 使用 double 累加，GPU 使用 FP32 GEMM，精度差异见架构文档。

## 各阶段对照

单位为 ms；加速比为 CPU / CUDA。除显存常驻特征聚合外，计时包含主机数据上传、设备运算、结果下载和临时分配。
特征分解、统计及 coreset 的主机输出也包括在计时内。

| 阶段 | 规模 | CPU | CUDA | 加速比 |
| --- | --- | ---: | ---: | ---: |
| 特征聚合（主机输入/输出） | 4x(512x56x56+256x28x28) | 101.4340 | 16.8594 | 6.02× |
| 特征聚合（网络输出已在显存） | same | 111.2930 | 2.6813 | 41.51× |
| 随机投影 | 4096x128_to32 | 11.7756 | 0.4841 | 24.32× |
| Coreset（包含投影） | 4096x128_K205_d32 | 29.4923 | 5.1908 | 5.68× |
| MH 白化 | 4096x128_R32 | 4.3233 | 1.1196 | 3.86× |
| SubAD 残差 | 4096x128_R32 | 7.2641 | 1.1130 | 6.53× |
| 均值/协方差统计 | 4096x128 | 18.3804 | 1.1298 | 16.27× |
| 协方差特征分解 | 128x128 | 32.9234 | 5.4023 | 6.09× |
| 预处理（含上传/下载） | 4x1080p_to512 | 8.7051 | 7.2144 | 1.21× |
| 异常图（含上传/下载） | 4x56x56_to512_sigma4 | 3.8564 | 2.2965 | 1.68× |
| 评分强制搬到 CUDA（反例） | 4x3136_K9 | 0.0893 | 0.1437 | 0.62× |

显存常驻聚合这一行比较两条实际数据路径：CPU 需要先下载网络特征再聚合，CUDA 直接消费显存特征并保留显存输出。
其约 41.51× 收益包括消除中转和布局处理，不能解释成同一 kernel 的纯计算加速比。
主机输入/输出聚合对照约 6.02×，更适合比较独立聚合阶段。

Coreset 的 CPU 测试将 `maxExactDistanceEvaluations` 设为 `SIZE_MAX`，确保双方都运行 greedy farthest-first，
没有将 CPU 的随机采样降级与 GPU 精确贪心过程混比。最终 bank 仍保存被选中的原始高维向量。

评分反例比较相同主机距离的 CPU 归约与强制上传后的 CUDA 归约，含有效性检查：GPU 更慢。
因此正式路径对 CPU 检索结果保持 CPU 评分，仅在距离已经位于显存时用 CUDA 评分。
小图预处理也不保证加速；下面的 224×224 实测即没有明显收益。

原始数据：[阶段 CSV](benchmarks/patchcore-gpu-stages.csv)。

## 实际网络端到端对照

使用项目自带 `wide_resnet50_2_patchcore.onnx`，输入为一张合成 224×224 图像，提取 layer2/layer3，
构造 1024×1536 的合成 memory bank，K=1、Gaussian sigma=4。
两个 TensorRT 模式使用相同的临时 engine；不修改原始模型文件。
CPU 模式使用 ONNX Runtime。模型加载、engine 构建不计入推理耗时。
每种模式预热 3 次，测量 9 次；按总耗时排序，报告中位样本的总耗时和各阶段耗时。

| 模式 | 总耗时 | 预处理 | 网络 | PatchCore adapter | 最终后处理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 66.4049 | 0.2799 | 37.9979 | 28.0728 | 0.0533 |
| TensorRT + CPU 计算阶段 | 25.6405 | 0.2669 | 3.4643 | 21.8193 | 0.0889 |
| TensorRT + CUDA 计算阶段 | 12.2145 | 0.2881 | 2.9026 | 8.8934 | 0.1289 |

本次 CUDA 计算链路相对原 TensorRT + CPU 阶段快 **2.10×**，总延迟减少 **52.36%**；
相对纯 CPU 模式快 **5.44×**。其中 adapter 从 21.8193 ms 降到 8.8934 ms，约 **2.45×**。
纯 CPU 对比同时包含 ONNX Runtime → TensorRT 的网络收益，不能全部归因于本次新增算子。

所有模式的 `faiss_provider=1`，即 CPU FAISS；最后一种模式的 `compute_provider=2`，即 CUDA compute。
本机没有 GPU Faiss SDK，因此新增显存检索扩展仅验证了模拟插件路由，**未完成真实 GPU Faiss 扩展的编译和执行验证**。
这些耗时并非全 GPU 检索的结果。

相同 TensorRT 网络输出、CPU/CUDA 计算阶段之间，raw score 绝对误差为 **0**，
raw anomaly map 最大绝对误差为 **2.67029e-5**（测试阈值 1e-3）。
该基准使用合成图像和 bank，验证计算一致性和耗时，不用于证明检测准确率。
原始数据：[推理 CSV](benchmarks/patchcore-gpu-inference.csv)。

## 测试结果与范围

- CUDA 构建 CTest：**20/20 通过**，含 12 个独立 GPU 测试和现有回归。
- 纯 CPU 构建 CTest：**7/7 通过**；关闭 TensorRT / CUDA compute / GPU Faiss 后，CPU 主库未引入 CUDA DLL 依赖。
- Compute Sanitizer memcheck：完整 GPU 测试套件 **0 errors**；最终预处理舍入调整后另外重跑预处理 memcheck，**0 errors**。
- 预处理与 OpenCV CPU 基线按 1e-6 浮点容差比较，覆盖上/下采样、单像素、灰度/BGR/BGRA、非连续输入、NCHW/NHWC 与多个插值模式。

| 测试 | 主要覆盖点 |
| --- | --- |
| aggregate | 多层池化、插值、拼接、布局、stride/padding、均值聚合与非法输入 |
| projection | 与 CPU double 累加基线对照投影数值 |
| coreset | CPU/GPU 选择、相同距离、重复向量、全部选取及非法输入 |
| statistics | 均值/协方差、特征分解、白化距离、SubAD 残差与非法输入 |
| preprocess | 缩放、裁剪、色彩、归一化、布局及非连续图像 |
| map | bilinear/Gaussian、REFLECT_101 边界与 CPU 数值对照 |
| score | CPU/GPU 距离归约、平方根、异常输入 |
| ownership | Tensor 设备检查、共享生命周期和跨线程释放 |
| fitting | MH/SubAD 拟合、bank 距离和 checkpoint |
| adapter | 插件加载、检索及 weighted score 的 CPU/GPU 对照 |
| backend | TensorRT 显存输入/输出、动态 batch、保留输出的生命周期 |
| session | 模式选择、执行信息、公开 C ABI、GPU fitter finalize 和整链路数值 |

这些测试覆盖本次修改涉及的算子、接口和资源管理；未在其他 GPU 架构、多 GPU 或 Linux 上执行。
本地详细日志位于 `build/patchcore-cuda/` 与 `build/patchcore-cpu-verify/`。
测试源码和新增 CMake 测试注册按要求未纳入提交；架构文档中的完整复测命令需要使用保留这些测试的本地工作区。
