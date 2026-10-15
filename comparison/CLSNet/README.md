# CLSNet comparison

官方代码：[HengYang01/CLSNet](https://github.com/HengYang01/CLSNet)。导入版本固定为
`6b13371297a78922c1d7fd3096fdf1e51f5f7192`，基于 `model/CLSNet.py`。
本目录为共享实验协议下的适配，采用模型名 **CLSNet**。

## 网络与官方实现的关系

保留官方 SCSC、HIGF、MSAB、3级编码/解码、特征残差连接及 L1 重建损失。
保留官方代码中已声明但未参与 forward 的模块，保证参数键与该基线实现一致。
按数据集选择固定宽度与注意力头数。CAVE与Houston13保留官方对应模型配置，
PaviaU、Chikusei、Botswana、Augsburg采用预先指定的跨数据集配置。
所有7个HIGF块中的42个MS_MSA模块均使用对应配置，输入输出波段数与hidden width分开设置。

| 数据集 | HSI波段 | MSI波段 | hidden channels | 注意力头数 | 每头维度 | 总参数量 |
|---|---:|---:|---:|---:|---:|---:|
| PaviaU | 103 | 4 | 120 | 4 | 30 | 15,032,292 |
| Houston13 | 144 | 8 | 150 | 5 | 30 | 23,440,964 |
| Chikusei | 128 | 8 | 150 | 5 | 30 | 23,397,764 |
| CAVE | 31 | 3 | 62 | 2 | 31 | 4,062,116 |
| Botswana | 145 | 8 | 150 | 5 | 30 | 23,443,664 |
| Augsburg synthetic / real Wald | 242 | 4 | 270 | 9 | 30 | 75,308,692 |

`hidden channels = 注意力头数 × 每头维度`。官方HIGF中名为`heads`的参数实际传入
每头维度；本版本明确区分`num_heads`与`dim_head`，避免将30维误写为30个头。
配置集中在`model.py`的`MODEL_PROFILES`，由训练、最终测试、固定形变测试和真实Wald
入口共用。数据集与实际HSI波段不匹配时立即拒绝，不自动缩窄网络。
这些配置延续官方网络的主要结构；它们并不代表相同参数预算，论文中需披露本表。
本目录不声称复现官方论文原训练配置或其已发表数值。

接口为 `model(lr_hsi, hr_msi) -> hr_hsi`，顺序与官方一致：

- 官方固定8×最近邻上采样改为可配置倍率，正式 synthetic 为4×、真实 Wald 为3×；
- 保留最近邻上采样；在上采样后对两路输入按相同尺寸补边到8的倍数，输出裁回原尺寸；
- 不增加外部配准网络、光谱校正头或新的损失项；
- 删除导入时执行的 CUDA/thop profiling，不要求 thop 或 timm。

CPU 对照验证：CAVE使用31+3通道，Houston使用官方144+4通道；在相同官方权重、
8×、16×24输出下，分别与`CLSNet.py`和`CLSNet_Houston.py`逐元素一致，最大绝对差均为0。
正式Houston13使用共享WV2的8波段MSI，输入卷积按8通道适配，内部仍为官方150维、5个头。
总参数量包含官方未参与forward的模块。

## 统一实验设置

请从仓库根目录执行。正式配置由 `experiment_protocol.py` 注入，
共享数据划分、SRF、退化与 EMR-Diff 一致：

| 项目 | 配置 |
|---|---|
| 数据集 | PaviaU / Houston13 / Chikusei / CAVE / Botswana / Augsburg synthetic |
| LR-HSI | physical，MTF@Nyquist=0.2，PSF truncate=3，4× |
| HR-MSI | 共享真实传感器 SRF，保持可靠坐标系 |
| 训练块 / 步长 / 评价块 | 64 / 32 / 128 |
| Stage1 | registered，800 epoch，seed=1001 |
| Stage2 | 600 epoch，seed=10，10% exact identity + 90% deformation |
| 形变 | dx,dy独立U(-4,4) HR px，旋转U(-2,2)°，局部幅度提议U(0,4)，5×5控制网格，min Jacobian=0.5 |
| 观测链 | `Y_H=P0(W_phi(X))`；HR-MSI与GT保持固定 |
| 优化器 / 学习率 / 权重衰减 | AdamW / 1e-5 / 5e-5，无cosine scheduler |
| batch | 1，沿用统一入口的通用方法默认值；EMR-Diff专属默认值为2 |
| 损失 | 官方 L1；真实 Wald 使用有效区域 masked L1 |
| 验证周期 | PaviaU=20，Houston13=10，Chikusei=5，CAVE=5，Botswana=20，Augsburg=10 |
| checkpoint选择 | Stage1验证PSNR；Stage2固定5个形变case的平均验证PSNR，min_delta=0.02 |
| 最终测试 | registered权重测Registered；mixed权重测Registered + Warp，10个形变case/patch |

Stage2只载入Stage1模型权重，重新建立优化器；正式混合分支没有额外的
全局α缩放。训练只读取训练和验证样本，不调用最终测试样本计算损失或选择epoch。
公共loader负责各数据集的完整空间/场景划分，具体以根目录README和`data/splits/`为准。

训练验证使用公共 `metrics.py`，最终报告使用与 EMR-Diff 最终测试完全相同的
公式（`final_metrics.py`）。最终结果对完整test split逐patch、逐case取算术均值；
每个patch重置到相同geometry seed（seed+70000）。RMSE保存原始[0,1]数值，
`RMSE_x255`只用于显示。固定形变测试直接调用公共`fixed_deformation_protocol.py`。

安装依赖：

```bash
python -m pip install -r comparison/CLSNet/requirements.txt
```

正式训练与测试，只需修改模型名、数据集名、阶段：

```bash
python train.py --model CLSNet --dataset PaviaU --mode registered
python train.py --model CLSNet --dataset PaviaU --mode mixed
python test.py --model CLSNet --dataset PaviaU --mode registered
python test.py --model CLSNet --dataset PaviaU --mode mixed

python test_fixed_deformation.py --model CLSNet --dataset PaviaU --mode mixed
```

继续训练到指定总epoch，恢复模型、AdamW、历史best、早停计数、Python/NumPy/
Torch/CUDA及形变generator状态；正式默认`num_workers=0`时支持确定性续训：

```bash
python train.py --model CLSNet --dataset PaviaU --mode registered --epochs 1600 --resume auto
python train.py --model CLSNet --dataset PaviaU --mode mixed --epochs 1200 --resume auto
```

checkpoint校验模型版本、数据集结构配置、hidden width、注意力头数、每头维度、
波段、数据集、阶段、退化、SRF指纹、patch及几何配置。
错误数据集、不同阶段或不同观测设置的权重会被拒绝。
结构版本为`clsnet_dataset_profiles_v2`；之前统一宽度62的v1权重不能用于本版
初始化、resume或测试，使用新配置重新训练。
产物均在本方法目录：

```text
checkpoints/physical/<Dataset>/{best,last}.pth.tar
checkpoints/hsi_warp_final/<Dataset>/{best,last}.pth.tar
logs/<family>/<Dataset>/validation_history.csv
outputs/unified_two_stage/<Dataset>/CLSNet_<mode>_seed10.json
outputs/fixed_deformation/<Dataset>/<mode>_seed10_cases10_eval1234/
```

每次启动会写入`checkpoints/.../run_protocol.txt`，记录结构配置、hidden width、注意力头数、每头维度、参数量、SRF、实际参数。

调试1轮（正式报告仍使用根入口协议）：

```bash
python comparison/CLSNet/train.py --dataset PaviaU --device cuda:0 --epochs 1 --validation_interval 1
```

保留旧常规退化兼容入口，仅用于对应协议的实验：

```bash
python comparison/CLSNet/train.py --dataset PaviaU --degradation_mode gaussian_bicubic
python comparison/CLSNet/test_hsi_deformed.py --dataset PaviaU --degradation_mode gaussian_bicubic --test_mode registered_only
```

两种退化的checkpoint分目录存储，测试会拒绝退化模式不一致的权重。

## Augsburg-2真实世界Wald

使用与 EMR-Diff 相同的 `Augsburg2-Wald-center-holdout-v1` 缓存、辐射标定、
禁入区、masked pooled指标及QNR实现。共享数据/QNR工具通过明确的
`comparison.UAFL`包路径复用，无需安装或加载UAFL网络。

- 242波段30m观测HSI提供Wald监督，真实Sentinel-2提供4波段MSI；
- 90m/30m训练，30m/10m推理，固定3×，没有10m HSI标签；
- 30m中心测试区`[24:72,36:84]`，训练禁入区含保护带`[18:78,30:90]`；
- train patch=24、stride=6、eval patch=48，100 epoch，每5 epoch验证；
- 最小验证SAM选择best；native推理仅重建未训练的144×144中心ROI，输出QNR/Dlambda/Ds；
- synthetic与real checkpoint的协议字段相互隔离，不允许混用。

```bash
python train_real.py --model CLSNet
python test_real.py --model CLSNet
```

缓存与标定文件位置沿用根入口配置。产物位于本目录的
`checkpoints/augsburg2_wald_center_holdout/`、`logs/augsburg2_wald_center_holdout/`
和`outputs/augsburg2_wald_center_holdout/`。

## 验证范围

```bash
python -m pytest -q tests/test_clsnet.py tests/test_clsnet_wald.py
```

测试包含6组数据集宽度/头数配置、全部42个注意力模块、3×/4×/8×、双路补边、真实forward/backward、
优化器/RNG续训恢复、跨阶段/跨数据集权重拒绝、两阶段训练与最终多patch形变测试。
测试还覆盖真实Wald入口的训练、参考测试、144×144中心ROI推理与QNR输出链路。
测试使用CPU和合成fixture，不代表已完成真实数据集训练或GPU显存测试。
