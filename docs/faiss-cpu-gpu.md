# Faiss CPU / GPU 检索

PatchCore、SPADE 的 Faiss 检索默认跟随最终选中的神经网络执行设备，使用同一份
CPU 序列化索引文件。选择发生在加载阶段，预测期间不会切换检索设备。

| 加载请求 | Faiss 行为 |
| --- | --- |
| CPU | CPU Faiss，不加载 GPU 插件，不探测 CUDA |
| GPU + FALLBACK_NONE | 必须成功初始化 GPU Faiss，否则加载失败 |
| GPU + FALLBACK_LOAD_ONLY | 优先 GPU Faiss，初始化不可用时可独立回退 CPU |
| AUTO | 网络选到 CUDA 时优先 GPU Faiss，可回退 CPU；网络选到 CPU 时直接 CPU Faiss |

Faiss 独立回退不会把已成功加载的神经网络也切换到 CPU。SPADE 所有层一起选择
GPU 或 CPU；中途某层初始化失败时先释放其他层已创建的 GPU 索引。文件损坏、
维度不匹配等模型错误不会被回退掩盖。预测时的搜索异常直接返回错误。

## 覆盖范围

- PatchCore 的主检索和加权评分的 support-neighbor 检索均使用选定设备。
  支持向量的 `reconstruct` 保留在 CPU，因此 CPU 索引会与 GPU 索引同时驻留。
- SPADE 每层独立持有索引与 GPU 资源，检索结果用于原有 CPU 异常图融合。
- GPU 索引加载时从 CPU 索引克隆一次，随后重复检索复用；不会每帧上传图库。
- 保持 Faiss FP32，不跟随 TensorRT 的 FP16 构建偏好降低检索精度。
- 默认 GPU 实现限制 `num_neighbors <= 2048`，只接受克隆结果为实际 `GpuIndex`
  的类型。工程生成的 `IndexFlatL2` 在支持范围内；CPU 包装索引/不支持的类型
  按加载回退策略处理，不能静默报告成 GPU。
- 每个 GPU 索引保留最多 64 MiB 配置的临时显存；查询分成最多 4096 条一组。
  索引本身和超出临时区的工作空间仍可能占用更多显存。
- 查询、距离和邻居编号跨插件时都是主机内存；因此仍有查询上传和结果下载。
  预处理、特征拼接/展平、后处理和训练侧索引构建仍在 CPU。

GPU 实例串行使用其资源，加载、搜索和销毁选择所属 CUDA 设备并恢复调用线程
之前的设备；返回前等待传输结束。调用方仍须保证 load/release 不与 predict
并发。并行工作线程使用不同算法对象。

## 构建与部署

`ANOM_ENABLE_FAISS_GPU` 默认 OFF，`*-cpu` 预设关闭，`*-nvidia` 预设开启。
也可以单独开启，例如在 ORT CUDA 推理方案下禁用 TensorRT、启用 GPU Faiss。

GPU Faiss 编译为独立的 `anom_search_faiss_cuda.dll`（Linux 为 `.so`），与算法
插件放在同一个 `plugin_directory_utf8` 目录，安装时一起打包。CPU 算法插件
只依赖原 CPU Faiss SDK，通过 C ABI 动态加载 GPU 检索插件。

需要准备与编译器/CUDA 匹配的 **静态、GPU-enabled Faiss SDK**：

```text
ANOM_FAISS_GPU_ROOT/
  include/faiss/gpu/StandardGpuResources.h
  lib/faiss.lib                 # Windows 静态库，不是 DLL 导入库
  lib/libfaiss.a                # Linux 静态库，需要 PIC
```

GPU 插件链接 CUDA Runtime、cuBLAS、OpenMP、BLAS 和 LAPACK。Faiss 应关闭 cuVS
以使用本构建配置；BLAS/LAPACK 可通过 CMake 的相应变量指定。不要将 GPU 版
`faiss.dll` 覆盖 CPU 部署的同名 DLL。独立静态链接用于避免这类名称/依赖冲突。
CUDA、cuBLAS、OpenMP、BLAS/LAPACK 的动态运行库仍需部署到加载器可见的位置。
只打开开关不会自动下载 SDK 或使 CPU Faiss 具备 GPU 能力。

## 执行信息

`get_execution_info` 保留原网络后端/provider 字段，并新增三个字段（使用原来
reserved 区域，结构大小和已有字段偏移不变）：

```c
/* info.struct_size = sizeof(info); object.get_execution_info(&object, &info); */
/* info.faiss_provider: ANOM_FAISS_NONE / ANOM_FAISS_CPU / ANOM_FAISS_CUDA */
/* info.faiss_device_id: CUDA device id，CPU 或无检索时为 -1 */
/* info.faiss_fallback_occurred: 是否独立从 GPU Faiss 回退到 CPU */
```

总的 `fallback_occurred` 同时包含网络与 Faiss 的回退；`fallback_reason_utf8`
包含具体失败原因。非 Faiss 算法报告 `ANOM_FAISS_NONE`。旧算法插件仍可在 CPU
模式加载；缺少新的 `anom_algorithm_query_execution_v1` 扩展时，严格 GPU 的
Faiss 请求被拒绝，允许回退的请求明确报告 CPU。

## 验证

不需要 SDK 的专项测试：

```sh
cmake -S tests/faiss_runtime -B build/faiss-routing
cmake --build build/faiss-routing --config Release
ctest --test-dir build/faiss-routing -C Release --output-on-failure
```

这套测试编译生产检索封装和 GPU 插件代码，但用测试替身模拟 Faiss/CUDA。
覆盖 CPU 无 GPU 依赖、缺插件/无设备、严格 GPU、加载回退、多个索引初始化
回滚、损坏文件不回退、加权评分支持检索、运行中不回退、跨线程生命周期。
它不能证明真实 GPU 的数值精度、性能或 SDK 链接兼容性。

完整 GPU 构建会额外注册 `faiss_gpu_parity_tests`，使用真实 Faiss/CUDA 对比
距离与邻居编号，并测试跨线程检索/销毁；只有设备探测失败时跳过，设备探测
成功后的索引加载、检索失败会使测试失败。业务模型还需进行误差容限、阈值
附近分类结果及端到端耗时对比；`adapter_ms` 包含 Faiss 检索及 CPU 特征处理。

实现依据 Faiss 官方的 [GPU 用法](https://github.com/facebookresearch/faiss/wiki/Faiss-on-the-GPU)、
[GPU 克隆接口](https://faiss.ai/cpp_api/namespace/namespacefaiss_1_1gpu.html) 和
[线程约束](https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls)。
