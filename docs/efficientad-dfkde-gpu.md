# EfficientAD / DFKDE GPU 推理

EfficientAD 和 DFKDE 复用可选 `anom_compute_cuda` 插件。网络特征保留在 GPU，
算法计算完成后下载最终异常图，保持原有公共 C ABI、输出格式和 CPU 路径。
CPU 构建无需 CUDA Toolkit，也不链接 CUDA 库。

## 全局设备策略

`InferenceSession` 根据算法插件协商出的 device extension 判断计算能力，
不再以 PatchCore/PaDiM 算法名单决定是否启用 CUDA。
当前支持的算法为 PatchCore、PaDiM、EfficientAD、DFKDE。

| 网络实际设备 / compute_mode | 行为 |
| --- | --- |
| CPU / AUTO 或 CPU | CPU 预处理、算法计算与后处理 |
| GPU / AUTO | 加载时尝试 CUDA compute；可重试的设备/插件错误允许 CPU 计算回退 |
| GPU / CPU | TensorRT 网络搭配 CPU 计算，作为对照基线 |
| GPU / CUDA | 必须成功启用该算法 CUDA 计算链路，否则加载失败 |
| 无设备扩展的算法 / CUDA | 明确返回不支持，避免忽略显式请求 |

```c
anom_algorithm_options_t options = {0};
options.struct_size = sizeof(options);
options.device = ANOM_DEVICE_GPU;
options.compute_mode = ANOM_COMPUTE_AUTO;
/* efficientad.load(&efficientad, package, &options); */
/* dfkde.load(&dfkde, package, &options); */
```

`compute_provider`、`device_tensor_path` 和回退原因继续反映实际路径。
自动回退只在加载阶段发生，运行中的 CUDA 错误直接返回。
旧的 host-only 算法插件可以继续使用 CPU 路径；旧计算插件不支持新能力探测时，
AUTO 可回退，显式 CUDA 报错。原有 compute v1 函数表布局没有改变。

此版本没有为 Direct、SPADE 或 YOLO 增加 CUDA compute，也没有将公共最终后处理迁入 GPU。
算法设备扩展的输出仍是主机异常图，不能将 `device_tensor_path` 理解为全流程没有下载。

## EfficientAD

数据流：

```text
CUDA 预处理 → TensorRT teacher / student / reconstruction_error
  → 通道 L2 归约 → 分别 resize → 每图每支 min/max 归一化
  → 权重融合 → Gaussian → 下载异常图 → 平滑后最大值
```

保留现有数学定义：

- hard 分支是 `sqrt(sum((teacher-student)^2))`，差值和累加使用 double。
- reconstruction 为单通道时原样使用，多通道时执行通道 L2。
- 两个分支可以具有不同空间尺寸；先 resize 到模型输入尺寸，再分别归一化。
- 常量图归零，融合权重除以权重总和，最终分数是平滑后异常图最大值。
- 不将 L2 改成 MSE，不改变单通道 reconstruction 的正负号。

同时修正两处数值问题：CPU adapter 与校准 pipeline 共用显式双精度
`(value-min)/(max-min)` 归一化，CUDA 使用同一公式，避免原缩放加偏移表达式
在大偏移、小范围数据上的消减误差和 SIMD 尾部差异。常量图保持零，非恒定图端点为 0/1，
不通过 epsilon 抹掉有效动态范围。共享 CUDA 双线性插值采用差值形式，保证常量图不产生舍入波动。

`gpu_efficientad.cpp` 提供通道归约、归一化与融合封装，
`cuda_efficientad.cuh` 实现设备算子；插值和 Gaussian 复用已有 CUDA map 实现。
模型加载时检查能力和 Gaussian 支持范围（sigma ≤ 511），避免预测开始后才发现配置不支持。

## DFKDE

数据流：

```text
CUDA 预处理 → TensorRT 多尺度特征 → 对齐与聚合
  → PCA 中心化及投影 → 全量图库精确 KDE → resize / Gaussian
  → 下载异常图 → 平滑后最大值
```

均值 `[D]`、投影基 `[D,K]` 和投影图库 `[G,K]` 在加载时上传一次并保持显存所有权。
特征聚合使用独立于 PatchCore 配置的通用接口。
PCA 保留 float 差值、double 乘加、float 投影结果；图库距离及 KDE 归约使用 double。

保持原有评分公式：

```text
score = min(distance_squared) / (2*sigma^2)
        - log(sum(exp(-(distance_squared-min(distance_squared)) / (2*sigma^2))))
```

GPU 仍遍历整个图库，不使用 top-k 近似，也不新增 `log(G)` 等归一化项。
按查询分块、图库分区计算稳定的最小距离及指数和，再合并分区统计，
避免分配完整 `[Q,G]` 距离矩阵。临时工作区默认预算 64 MiB，不包含输入、输出和常驻模型参数。
小于单个查询所需工作区的配置明确报错。

`gpu_dfkde.cpp` 负责形状和参数验证，`cuda_dfkde.cuh` 负责投影、分区 KDE 与归约。
加载阶段创建上下文、上传常驻参数时的显存不足，以及插件不支持和可预判的参数限制，按设备策略处理。
推理时临时缓冲分配失败属于运行期错误，直接返回。
模型整体不能驻留显存时，AUTO 可回退；本次没有实现把常驻图库逐次从主机流入 GPU 的模式。

## DFKDE CPU 拟合修复

OpenCV PCA 返回按行存放的 `[K,D]` 主成分，而既有 DFKDE v1 读取接口使用 `[D,K]` 投影基。
拟合现将主成分转置后用于投影及序列化，修复 `K<D` 时的矩阵维度错误。
文件版本保持 v1，读取布局不变；现有符合 `[D,K]` 约定的模型无需转换。
新增多维降维、产物读取和 CPU 推理基线测试覆盖这一修复。

本次 GPU 工作范围是推理；没有新增 EfficientAD 网络训练或 DFKDE 公开 GPU fitter。

## 构建与验证

配置项目依赖后启用 `ANOM_ENABLE_TENSORRT=ON`、`ANOM_ENABLE_CUDA_COMPUTE=ON`。
计算插件需要 CUDA Toolkit、cuBLAS/cuSOLVER；两种算法均不需要 GPU Faiss。

```sh
cmake -S . -B build/cuda -G Ninja -DCMAKE_BUILD_TYPE=Release -DANOM_ENABLE_TENSORRT=ON -DANOM_ENABLE_CUDA_COMPUTE=ON -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build/cuda
ctest --test-dir build/cuda --output-on-failure
```

本地 `other_cuda_tests` 覆盖算子、adapter、全局模式、公开 C ABI 和 PCA 拟合基线，
并提供 `--benchmark` 与 `--real-benchmark`。测试源码及相应 CMake 注册保留本地。
DFKDE 端到端基准使用现有 WideResNet 特征网络和合成统计；EfficientAD 使用合成特征对比实际 adapter，
没有将合成流路由测试当作真实 EfficientAD 网络性能。

性能条件、误差与结果见 [性能报告](efficientad-dfkde-gpu-benchmark.md)。
