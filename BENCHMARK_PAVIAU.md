# PaviaU 复杂度汇总

两个仓库放在同一级目录，在任意一个仓库的根目录运行：

```bash
python benchmark_paviau.py --device cuda:0
```

两个入口生成同一份表：`comparison_experiments/outputs/benchmark_paviau/summary.csv`。
每完成一个方法就写入结果；单个权重缺失或推理失败会在表中记录原因，其他方法继续。
各方法的执行日志位于表格旁的 `logs/`。可用 `--dry_run` 查看计划，
或用 `--methods CNMF HySure ours --output_dir /path/to/smoke` 单独检查部分方法。

权重不在同一台机器时，可在装好模型依赖且有 PaviaU 数据的服务器上运行：

```bash
python benchmark_paviau.py --device cuda:0 --no_checkpoints --emr_width 128
```

该模式用固定种子和各模型自身的初始化方式执行完整推理，无需 `.pth` 文件。
默认输出到 `outputs/benchmark_paviau_initialized/summary.csv`，表中标明
`weight_source=initialized`。EMR-Diff 使用指定的骨干宽度（这里为 128），
其他基线采用当前 PaviaU 标准架构。若 ours 的论文配置不同于默认值，
用末尾的 `--ours_args` 指定与正式实验相同的宽度和迭代步数。
同架构时，参数量与权重数值无关；FLOPs 和耗时对应实际跑过的路径。
基于输入或权重动态选路的算子，可能在训练后执行不同的运算或耗时。

### 数据、配置与权重

唯一输入为同一张 PaviaU 中心留出测试裁块生成的配准观测。批大小为 1，
LR-HSI 为 103×32×32，HR-MSI 为 4×128×128，输出为 103×128×128。
空间退化为物理 ×4、MTF 0.2、PSF truncate 3.0，MSI 由 IKONOS 四波段 SRF 生成。
所有方法共享同一对输入及 SRF，表中的 `input_sha256` 用于核对。
`PaviaU.mat` 默认从两个仓库的 `data/raw` 查找，也可传 `--data_root /path/to/raw`。

- MoG-DCN、PSRT、ESFS、CLSNet、EMR-Diff 默认读 `checkpoints/physical/PaviaU/`；
  M2DTN、PMI-RFCoNet、MoE-PNP、SA-RFM、UAFL 默认读 `checkpoints/hsi_warp_final/PaviaU/`。
  两组均使用相应主实验权重，输入始终为同一对配准观测。
- EMR-Diff 默认读 `best.pth.tar`；若没有默认权重但只有一种 `best_base*.pth.tar`
  则使用该宽度。用 `--emr_width 128` 可直接选 `best_base128.pth.tar`。
- 用 `--checkpoint METHOD=/path/to/best.pth.tar` 覆盖某方法路径。
  若论文使用的是另一训练阶段，还需如实指定 `--model_stage METHOD=registered`
  或 `--model_stage METHOD=mixed`；路径和训练阶段都记录在表中。
- ours 使用完整 CDRDI → Physics-Grounded Diffusion → HPGC-Refiner 推理链，
  即使输入已配准也运行 CDRDI。默认按 S2Diff-MH 的 PaviaU 最终测试设置查找权重，
  包括通过 refiner 检查点元数据解析 geometry 和 diffusion 权重。
  自定义配置用 `--ours_args` 透传到 S2Diff-MH，并放在命令最后。

```bash
python benchmark_paviau.py --device cuda:0 --emr_width 128 \
  --ours_args \
  --refiner_checkpoint /path/to/PaviaU_refiner.pth \
  --geometry_checkpoint /path/to/PaviaU_geometry.pth \
  --diffusion_checkpoint /path/to/PaviaU_diffusion.pth
```

如果仓库目录名不同，从本仓库用 `--s2diff_root /path/to/S2Diff-MH`，
或从 S2Diff-MH 用 `--comparison_root /path/to/comparison_experiments`。

### 指标含义

| 列 | 定义 |
| --- | --- |
| `params_M` | 去重后的模型参数量（百万）。ours 计入推理用到的三个训练模块，包括冻结的权重；固定传感器算子不作为参数。CNMF、HySure 的学习参数量为 0。 |
| `FLOPs_G` | 每次完整重建的可计运算量（十亿）。采用 1 MAC = 2 FLOPs，统计 PyTorch 能识别的卷积、矩阵乘、注意力，以及可变形卷积的加权乘加。递归、扩散和 ODE 中实际重复执行的运算会重复计数。它是近似值：FFT、采样插值、逐元素运算及无计数公式的自定义算子不计，主要未统计算子见 `flops_uncounted_ops`。CNMF、HySure 不套用神经网络的 FLOPs 计数，留空。 |
| `latency_ms_median` | 单对观测推理中位耗时（毫秒），同时保存均值和标准差。NN 以 FP32、TF32 关闭、无梯度运行；CUDA 每次计时都同步。只计从已就绪的输入到重建输出；不计数据加载、权重读取、输入传输、指标计算及文件保存。 |

神经方法默认预热 3 次、测量 20 次；CNMF、HySure 在 CPU 用正式求解配置完整求解，
默认单线程、预热 1 次、测量 3 次。使用 `--warmup`、`--repeats`、
`--classical_warmup`、`--classical_repeats` 和 `--cpu_threads` 可调整。
论文中标注传统方法在 CPU 求解，神经方法按表中记录的设备报告。
