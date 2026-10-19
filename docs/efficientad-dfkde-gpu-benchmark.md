# EfficientAD / DFKDE GPU 性能与验证

测量日期：2026-10-20。实现与设备策略见 [GPU 推理设计](efficientad-dfkde-gpu.md)。
两种算法均可随全局 GPU + AUTO 启用 CUDA compute，无需 GPU Faiss。
以下结果针对本机和所列规模；本次没有修改 EfficientAD 网络训练，也没有新增 DFKDE GPU 拟合。

## 环境与测量方法

- Intel Core i5-13400F（10 核、16 逻辑处理器），RTX 5070 Ti 16 GB，Windows。
- 驱动 616.64，CUDA 13.2.78，计算架构 120；TensorRT 10.16.1.11、OpenCV 4.13。
- MSVC 19.51.36257，Release / Ninja；启用 ORT、TensorRT、CUDA compute，关闭 GPU Faiss。
- 阶段基准预热 2 次，取 7 次中位数；GPU 调用同步完成后结束计时。
- 使用库默认线程配置，未锁 GPU 频率、CPU 亲和性；计时期间没有并行编译、其他测试或 GPU 检查。
- CPU adapter 基线是实际 SDK 实现；单独 KDE 算子基线为测试内等价的 double 累加参考循环，未额外优化 CPU SIMD/GEMM。
- 数据是一次基准运行中重复测量的中位数，不含跨进程置信区间；不同 GPU 边界分别计时，同规模 KDE 的 CPU 参考复用同一中位数，不能相减推算传输耗时。

## 实际网络端到端：DFKDE

采用项目自带 `deployment/model/wide_resnet50_2_patchcore.onnx`，输入一张合成 224×224 BGR 图像。
提取 layer2（512×28×28）与 layer3（1024×14×14），聚合到 784×1536，
使用合成 PCA 均值、16 维投影基及 1024 条图库，kernel sigma=1.7、Gaussian sigma=4。
导出模型原有的分类输出未裁剪。

CPU 使用 ONNX Runtime；两种 TensorRT 模式共用同一个新建的临时 engine。
加载、统计上传、engine 构建不计入稳态耗时。各模式预热 3 次、测量 9 次，
按总耗时取中位样本及其阶段耗时。单位为 ms。

| 模式 | 总耗时 | 预处理 | 网络及传输 | DFKDE adapter | 最终后处理 |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU | 81.3002 | 0.2122 | 39.2879 | 41.7393 | 0.0600 |
| TensorRT + CPU 计算 | 37.9134 | 0.2796 | 3.7460 | 33.8229 | 0.0640 |
| TensorRT + CUDA 计算 | 5.2250 | 0.1545 | 3.0560 | 1.9706 | 0.0434 |

新增 CUDA 链路相对同一 TensorRT 网络搭配 CPU 计算快 **7.26×**，总延迟减少 **86.22%**；
adapter 阶段快 **17.16×**。相对纯 CPU 的 **15.56×** 包含网络切换到 TensorRT 的收益。
网络阶段也包含不再下载完整特征的收益，最终异常图仍需下载。

同一 TensorRT engine 下，CPU/CUDA 计算的 raw score 绝对误差为 **0**，
raw anomaly map 最大绝对误差为 **2.86102e-6**，均通过 1e-3 阈值。
CUDA 模式实际遥测为 `compute_provider=2`、`device_tensor_path=1`。
该比较没有混入 ORT 与 TensorRT 的网络误差。

图像、PCA 参数和图库为合成数据；这验证了计算与性能，没有测量真实数据集的 AUROC。
原始数据：[端到端 CSV](benchmarks/efficientad-dfkde-gpu-inference.csv)。

## 实际 adapter 与算子

EfficientAD 输入为 batch=4、teacher/student 各 384×56×56，单通道 reconstruction 为 28×28，
hard/soft 权重为 0.7/2.3，输出 224×224，Gaussian sigma=4。
DFKDE adapter 使用与上表相同的特征形状和统计规模，但网络输出由合成张量替代。

两类边界分别表示：

- **主机特征起点**：CPU 已持有特征；GPU 计入上传和最终异常图下载。
- **显存特征起点**：模拟 GPU 网络输出；CPU 计入完整特征下载，GPU 直接消费显存特征；两者均返回主机异常图。

| 实际 adapter / 起点 | CPU ms | CUDA ms | 加速比 |
| --- | ---: | ---: | ---: |
| EfficientAD / 主机特征 | 12.4774 | 6.0355 | 2.07× |
| EfficientAD / 显存特征 | 25.4888 | 1.5109 | 16.87× |
| DFKDE / 主机特征 | 36.7602 | 2.2583 | 16.28× |
| DFKDE / 显存特征 | 36.2852 | 2.0115 | 18.04× |

EfficientAD 暂无完整三路网络模型，因此 **16.87× 是 adapter 边界的收益，不是完整网络端到端提速**。
较大的收益包含避免传回两组高维特征。主机特征模式显示传输会明显影响收益。

单独 KDE 的 PCA 均值、投影基及图库预先常驻显存。Q=784、K=16；主机查询模式计入查询上传与分数下载，
显存查询模式只测设备算子，结果保持显存。CPU 参考始终在主机计算。

| 查询维度 D / 图库 G / 起点 | CPU ms | CUDA ms | 加速比 |
| --- | ---: | ---: | ---: |
| 128 / 4096 / 主机 | 53.1367 | 1.2298 | 43.21× |
| 128 / 4096 / 显存 | 53.1367 | 1.2233 | 43.44× |
| 1536 / 1024 / 主机 | 39.3249 | 1.5792 | 24.90× |
| 1536 / 1024 / 显存 | 39.3249 | 0.7739 | 50.81× |

这些算子比值不能作为完整 SDK 的加速比。全部阶段原始数据，包括通道 L2 单算子，见
[阶段 CSV](benchmarks/efficientad-dfkde-gpu-stages.csv)。
其中 L2 的 `device_boundary` 行在 CPU 侧计入两组特征下载，GPU 输出保持显存；
该行 181.25× 是计算边界对比，不能解释为单纯 CUDA kernel 相对 CPU 算子的加速比。

## 验证结果

- GPU 构建 CTest **32/32**，纯 CPU 构建 **8/8**；包括已有 PatchCore/PaDiM 回归。
- EfficientAD 通道 L2、单/多通道 reconstruction、不同尺寸、多 batch、归一化融合、Gaussian、最终分数均覆盖。
- 归一化覆盖高偏移小范围图、3/5/17 像素尾部、±FLT_MAX；共享插值覆盖正负常量图，要求无平滑时逐元素保持常量。
- DFKDE 覆盖多尺度聚合、PCA、全图库 KDE、不同带宽、128 字节工作区强制分块、主机/显存输入及非法形状参数。
- adapter 异常图与分数使用 `3e-5 + 5e-6*abs(reference)` 容差；归一化边界使用更严格的 `3e-7 + 3e-6*abs(reference)`，并对端点精确断言。
- 两种算法均验证全局 GPU/AUTO、显式 CUDA、强制 CPU、不可用设备回退、公开 C ABI，以及不支持的 Gaussian 在加载阶段回退。
- DFKDE CPU 拟合验证 PCA 转置后的 `[D,K]` 序列化、产物读取与推理基线。
- Compute Sanitizer memcheck：EfficientAD/DFKDE 算子及两种 adapter 共 **4 个 case，均为 0 errors**。
- CPU 构建的主库及两个算法插件经依赖检查，无 CUDA/TensorRT 导入；Faiss/OpenCV 依赖亦无此类导入。

本地复现入口（测试源码及其 CMake 注册未提交）：

```powershell
ctest --test-dir build/patchcore-cuda --output-on-failure
ctest --test-dir build/patchcore-cpu-verify --output-on-failure
./build/patchcore-cuda/bin/other_cuda_tests.exe --benchmark
./build/patchcore-cuda/bin/other_cuda_tests.exe --real-benchmark
```

加载时常驻参数上传失败可按 AUTO 策略回退；预测时临时缓冲不足属于运行期错误。
没有实现图库分批从主机流入显存，也没有把公共最终后处理移入 GPU。
