# EMR-Diff Comparison

本目录保存 EMR-Diff 对比实验代码及其统一实验协议适配。EMR-Diff 通过 `dataset_loader/ufg_adapter.py` 复用仓库根目录公共数据、SRF、退化和评价指标接口。

## Shared comparison protocol

### MSI 模拟固定

```text
PaviaU:
  HR-MSI = IKONOS SRF
  Blue / Green / Red / NIR
  MSI channels = 4

Houston13:
  HR-MSI = WorldView-2 all8 SRF
  MSI channels = 8

Chikusei:
  HR-MSI = WorldView-2 all8 SRF
  MSI channels = 8
```

EMR-Diff 不再使用 uniform band selection。

### LR-HSI 支持两种退化模式

所有正式对比方法都必须支持以下两种共享退化模式：

```text
gaussian_bicubic:
  Gaussian kernel = 5x5
  sigma = 2.0
  bicubic downsampling x4

physical:
  MTF at LR Nyquist = 0.2
  MTF-derived Gaussian optical PSF
  detector pixel-area integration
  stride sampling x4
  PSF truncate = 3.0
```

默认模式是 `gaussian_bicubic`。物理退化通过 `--degradation_mode physical` 显式启用。两种模式只改变 LR-HSI 观测模型，不改变 HR-MSI 的 IKONOS/WV2 SRF 协议。

### Train / validation / test 空间划分

```text
scale factor = x4
train patch = 64x64
stride = 32
validation region = fixed 128x128 region, disjoint from test
final test region = center 128x128
```

PaviaU 保持既定的左上角 128x128 验证区。Chikusei 由于整幅场景远大于 PaviaU，极端左上角单块可能出现低能量或低动态范围，从而使 validation PSNR 虚高，因此固定改为左上象限内部的 128x128 验证区；位置只由图像尺寸决定，不依据模型结果选择。最终测试区仍固定为中心 128x128，训练 patch 同时避开验证区与最终测试区。

数据加载时会额外打印验证块的 `min / mean / max / std`，用于检查验证区是否异常接近零或缺乏动态范围。

训练阶段只使用训练集与验证集。最终中心 128x128 测试区不用于 epoch 选择、调参或 early stopping，避免测试泄漏。

动态状态通道数：

```text
PaviaU:    103 HSI + 4 MSI = 107 channels
Chikusei:  128 HSI + 8 MSI = 136 channels
```

Houston13 按实际 HSI 波段数加 8 个 WV2 MSI 通道自动确定。

### Metric consistency

公共 `metrics.py` 对预测值与 GT 统一在 `[0,1]` 数值域计算 PSNR、RMSE、SAM、ERGAS、SSIM 和 CC，不再出现 PSNR/RMSE 使用 clamp 而 SAM/ERGAS/SSIM/CC 使用未裁剪预测值的情况。

SAM 使用有效非零光谱像素上的标准余弦夹角计算。旧实现把 `eps` 同时加入两个光谱范数和最终分母，在 Chikusei 近零光谱区域会把余弦值人为压低，使几乎相同的低幅值光谱也可能得到几十度的 SAM。该数值不稳定问题已经修复。

## Early stopping

EMR-Diff 使用独立验证区 PSNR 进行早停。考虑不同数据集单个 epoch 的计算成本差异，验证间隔按数据集设置，而不是固定每 100 epoch 才评估一次：

| Dataset | Validation interval | Max delay to first validation after convergence |
|---|---:|---:|
| PaviaU | 20 epochs | <20 epochs |
| Houston13 | 10 epochs | <10 epochs |
| Chikusei | 5 epochs | <5 epochs |

默认早停参数：

```text
max epochs = 3000
monitor = PSNR
min_delta = 0.02 dB
patience = 2 validation evaluations
eval_seed = 1234
```

因此，在默认设置下，从某个最佳点开始连续没有有效提升时，额外训练上限约为：

```text
PaviaU:    2 x 20 = 40 epochs
Houston13: 2 x 10 = 20 epochs
Chikusei:  2 x 5  = 10 epochs
```

只有当验证 PSNR 比历史最佳至少提高 0.02 dB 时才计为有效提升。连续 2 次验证均无有效提升时停止训练。

由于 EMR-Diff 推理包含扩散随机噪声，验证与最终测试固定 `eval_seed=1234`，确保同一 checkpoint 的评价可复现，避免随机波动干扰 early stopping。

每次出现新的最佳验证 PSNR，会覆盖保存：

```text
comparison/EMR-Diff/checkpoints/<degradation_mode>/<Dataset>/best.pth.tar
```

每次正常验证仍会保存对应 epoch checkpoint：

```text
model_epoch_<N>.pth.tar
```

验证历史写入：

```text
comparison/EMR-Diff/logs/<degradation_mode>/<Dataset>/validation_history.csv
```

正式 `Test.py` 在不指定 `--checkpoint` 时优先加载 `best.pth.tar`。

### 手动覆盖验证间隔

需要临时调整时使用：

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic \
  --validation_interval 10
```

正式默认值仍建议保持数据集统一：PaviaU=20、Houston13=10、Chikusei=5，避免同一数据集不同方法采用不同验证频率。

### Early stopping 参数覆盖

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic \
  --early_stop_patience 2 \
  --early_stop_min_delta 0.02 \
  --early_stop_metric PSNR \
  --eval_seed 1234
```

将 `--early_stop_patience 0` 可关闭 early stopping，但正式对比实验不建议这样做。

## Train

### 常规退化

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic

python comparison/EMR-Diff/Train.py \
  --dataset Chikusei \
  --degradation_mode gaussian_bicubic
```

### 物理退化

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode physical

python comparison/EMR-Diff/Train.py \
  --dataset Chikusei \
  --degradation_mode physical
```

训练启动后，在真正产生第一个 epoch checkpoint 之前，程序就会先创建当前协议目录并写入：

```text
comparison/EMR-Diff/checkpoints/<degradation_mode>/<Dataset>/run_protocol.txt
```

该文件记录 `requested_dataset`、`resolved_dataset`、`requested_degradation_mode`、`resolved_degradation_mode`、验证间隔和 early stopping 参数。这样无需等待第一个 validation epoch，即可确认当前运行对应的数据集、退化模式和 checkpoint 目录。

### 1轮 smoke test

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic \
  --epochs 1 \
  --validation_interval 1

python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --epochs 1 \
  --validation_interval 1
```

## Test

测试时 `--degradation_mode` 必须与训练 checkpoint 的退化模式一致。默认优先加载 `best.pth.tar`：

```bash
python comparison/EMR-Diff/Test.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic

python comparison/EMR-Diff/Test.py \
  --dataset PaviaU \
  --degradation_mode physical
```

## Experiment storage rule

两种退化模式的实验产物完全隔离：

```text
comparison/EMR-Diff/checkpoints/<degradation_mode>/<Dataset>/
comparison/EMR-Diff/logs/<degradation_mode>/<Dataset>/
comparison/EMR-Diff/outputs/<degradation_mode>/<Dataset>/
```

`metrics.txt` 只用于最终测试；`validation_metrics.txt` 用于训练期验证。

## Important protocol note

在引入独立验证区之前，旧版 EMR-Diff 训练代码把中心 128x128 区域既作为训练期评估区域又作为最终测试区域。该旧流程会造成测试集参与 epoch 选择，因此旧版 100/200/300 epoch 评估结果只能用于调试与趋势观察，不应直接作为采用新版协议后的正式最终测试结果。

Chikusei 早期新版实验若使用极端左上角 128x128 验证块，并出现类似 `PSNR≈60 dB` 与 `SAM≈50°` 的矛盾组合，也只作为诊断结果，不用于正式模型选择。正式 Chikusei 实验应使用当前 interior validation 规则及修正后的公共 metrics。

新版代码已经将验证区与中心测试区完全分离，并采用数据集自适应验证频率。正式实验以 `best.pth.tar` 在最终测试区上执行正式测试。

## Shared implementation

LR-HSI 不由 EMR-Diff 自己重新实现，而是统一调用仓库根目录 `degradations/` 与 `data_loader.py`。这样 EMR-Diff 与后续加入的其他对比方法使用完全相同的 `gaussian_bicubic` 和 `physical` 观测模型。

## Dependencies

EMR-Diff 依赖 PyTorch、OmegaConf、SciPy、tqdm、timm。公共退化模块由 PyTorch 实现，不再依赖 OpenCV 才能得到正式退化结果。
