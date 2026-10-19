# PaDiM GPU 模式设计

PaDiM 复用可选 `anom_compute_cuda` 插件，不需要 GPU Faiss。CPU 构建仍不链接 CUDA。
全局设备选择、CUDA 计算模式、插件加载与回退沿用 PatchCore 的公共约定。

## 设备选择与兼容

- CPU 网络：PaDiM 计算在 CPU；显式要求 CUDA compute 时返回参数错误。
- GPU 网络 + `ANOM_COMPUTE_AUTO`：尝试 CUDA compute，加载时不可用可回退 CPU，并记录执行信息。
- GPU 网络 + `ANOM_COMPUTE_CPU`：使用 CPU 计算，便于与同一 TensorRT engine 做对照。
- GPU 网络 + `ANOM_COMPUTE_CUDA`：必须成功加载计算插件并上传统计参数，失败直接返回错误。

算法设备扩展增加 PaDiM 支持，旧 host-only ABI 不变。计算插件通过 PaDiM capability 操作检测
新增算子，防止旧插件在模型加载成功后才发现缺少操作。`compute_provider` 和
`device_tensor_path` 表示实际选择，PaDiM 的 `faiss_provider` 始终为无检索阶段。

拟合选项 `anom_padim_fitter_options_t.compute_mode` 使用一个原 reserved 字段，结构大小与其余
已有字段偏移不变。AUTO 跟随模板的后端；拟合期间计算错误不会重新执行 CPU。

## 推理数据流

```text
CPU 图像 → CUDA 预处理 → TensorRT 显存特征
  → 插值与选定通道提取 → 逐位置马氏距离 → 插值/Gaussian → CPU 输出
```

主机端 `GpuCompute::selectFeatures` 是通用特征操作，不依赖 PatchCore 配置。
它将多尺度特征直接写成 `[N,H,W,D]`，只处理选定通道，避免分配全部通道的拼接张量。
通道次序、bilinear 对齐和第一层特征网格与现有 CPU 实现一致。

模型加载时将 `mean[L,D]`、`precision[L,D,D]` 上传一次并持有显存所有权。
推理按位置并行计算 `sqrt(max((x-mean)^T precision (x-mean), 0))`，
保留 float 差值、double 累加以及非法二次型检查，不默认启用低精度计算。

异常图复用已有 CUDA 插值和可分离 Gaussian 算子。输出需要返回主机，因此下载后在 CPU 取最大值；
这保持 PaDiM 的“平滑后异常图最大值”评分定义，不能替换成原始 patch 距离最大值。
最终坐标恢复、阈值、区域分析等通用后处理保持原实现。
CUDA Gaussian 当前支持 sigma ≤ 511；更大的合法 CPU 配置在加载阶段由 AUTO 回退，
显式 CUDA 模式返回错误，不等到首次预测才发现不支持。

## 拟合数据流与显存预算

按空间位置累计 double `sum` 和 `sum(x x^T)`，保持现有总体协方差公式：

```text
mean = sum / N
covariance = secondMoment / N - mean * mean^T + regularization * I
```

现有 PatchCore-MH 的全局 covariance 算子不适合此处；PaDiM 的每个空间位置均有独立矩阵。
训练统计状态小于默认 512 MiB 预算时驻留 GPU；超过预算时保留主机状态，按位置分块上传和更新。
CPU-backed 分块限制设备工作集，但增加 PCIe 传输，不能保证每批都比 CPU 快。
预算是统计状态/分块策略的控制量，不是整个进程或 TensorRT 的显存硬上限。

finalize 分块构建协方差、执行 double Cholesky 与三角求解，导出原有 float32 均值和精度矩阵。
每个位置检查非正定与非有限结果，保持 `ANPADIM` v1 和 checkpoint 格式兼容。
GPU 统计在保存 checkpoint 时下载，CPU checkpoint 可继续在 GPU 上训练，反向亦可。

以 56×56 网格为例，仅 FP32 精度矩阵在 D=100 时约 120 MiB，D=550 时约 3.53 GiB；
训练 FP64 二阶矩占用翻倍。分块求解避免额外同时保留全部位置的 covariance、factor 和 inverse 临时矩阵。

## 实现边界

- `src/compute/gpu_padim.cpp`：参数验证、设备缓冲和操作调用。
- `src/compute/cuda_padim.cuh`：特征选择、距离、统计更新与批量求解。
- `src/adapters/padim_adapter.*`：统计参数常驻显存和推理链路。
- `src/pipeline/padim.*`：训练状态、显存预算、checkpoint 和产物写入。
- `src/model/inference_session.cpp`、`src/c_api/model_management.cpp`：运行时与拟合模式路由。

各调用返回前完成计算，跨插件继续使用显式同步合约；此版本不引入跨插件 stream/event 或 CUDA Graph。
固定统计参数只读共享，设备缓冲保留插件生命周期。CPU/GPU 并行归约次序不同，结果按误差容限比较，
不要求所有浮点位相同。

## 验证与性能分析

本地测试覆盖选定通道顺序和非整数缩放、完整非对角精度矩阵、非法输入、平滑后评分、
CPU/GPU 拟合产物、驻留与分块统计、checkpoint 双向恢复，以及公开 C ABI 设备选择。
基准分别测量算子、拟合、adapter 和实际网络端到端，区分主机输入/输出与显存常驻路径。
测量前预热，报告重复测量中位数；模型加载、统计上传、engine 构建不计入稳态推理耗时。
所有性能结论以本机实际测量为准，不预设加速倍数。

实际数据、测量边界和误差见 [PaDiM GPU 性能报告](padim-gpu-benchmark.md)。

## 使用与复现

推理使用现有公共选项；全局选择 GPU 后，默认 AUTO 即会尝试 PaDiM CUDA 计算。
需要确保实际启用时使用显式 CUDA，并检查返回状态及 `get_execution_info`。

```c
anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
options.device = ANOM_DEVICE_GPU;
options.compute_mode = ANOM_COMPUTE_CUDA;
/* padim.load(&padim, package_path, &options); */
```

拟合通过 `anom_padim_fitter_options_t.compute_mode` 选择计算模式，网络设备由模板配置决定。
需要 TensorRT 拟合模板才能显式选择 CUDA。当前 512 MiB 统计预算是内部 C++ 配置，
没有新增公开 C ABI 显存预算参数。

需要 CUDA Toolkit、兼容的宿主编译器、cuBLAS/cuSOLVER，以及 TensorRT 和项目常规依赖。
例如在 Windows 的 MSVC 开发者终端，配置相应依赖路径后执行：

```sh
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DANOM_ENABLE_TENSORRT=ON -DANOM_ENABLE_CUDA_COMPUTE=ON -DANOM_ENABLE_FAISS_GPU=OFF -DCMAKE_CUDA_ARCHITECTURES=native -DBUILD_TESTING=ON
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
```

本地 `padim_cuda_tests` 有 6 项独立 CTest：selection、distance、statistics、fitting、adapter、session。
session 需要 TensorRT；其他 5 项可单独验证计算插件。
按要求，本次提交不包含新增或修改的测试源码，也不包含新增 CUDA 测试目标的 CMake 注册。
干净检出只会运行已提交的测试；以下命令需要本地保留的测试源码及其 CMake 注册：

```sh
ctest --test-dir build/cuda -L padim --output-on-failure
build/cuda/bin/padim_cuda_tests --case fittingTest
build/cuda/bin/padim_cuda_tests --benchmark
build/cuda/bin/padim_cuda_tests --real-benchmark
compute-sanitizer --tool memcheck --error-exitcode 1 build/cuda/bin/padim_cuda_tests --case statisticsTest
```

Windows 可执行文件带 `.exe` 后缀；多配置生成器使用 `bin/Release`。
实际网络基准需要 `deployment/model/wide_resnet50_2_patchcore.onnx`，在临时模型包内构建 engine，
不修改原始 ONNX。测试和基准源码保留在本地工作区。
