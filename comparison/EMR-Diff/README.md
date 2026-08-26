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

默认模式是 `gaussian_bicubic`。物理退化通过 `--degradation_mode physical` 显式启用。

两种退化模式只改变 LR-HSI 观测模型，不改变 HR-MSI 的 IKONOS/WV2 SRF 协议。

### Train / validation / test 空间划分

```text
scale factor = x4
train patch = 64x64
stride = 32
validation region = fixed 128x128 region, disjoint from test
final test region = center 128x128
```

验证区优先取图像左上角 128x128；若与中心测试区重叠，公共数据加载器自动尝试其他角落。所有训练 patch 同时避开验证区与最终测试区。

训练阶段只使用训练集与验证集。最终中心 128x128 测试区不再用于 epoch 选择、调参或 early stopping，避免测试泄漏。

动态状态通道数：

```text
PaviaU:    103 HSI + 4 MSI = 107 channels
Chikusei:  128 HSI + 8 MSI = 136 channels
```

Houston13 按实际 HSI 波段数加 8 个 WV2 MSI 通道自动确定。

原始 HSI 数据统一放在：

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

评价指标统一调用仓库根目录 `metrics.py`。

## Early stopping

EMR-Diff 默认使用独立验证区 PSNR 进行早停：

```text
max epochs = 3000
validation interval = 100 epochs
monitor = PSNR
min_delta = 0.02 dB
patience = 2 validation evaluations
eval_seed = 1234
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

正式 `Test.py` 在不指定 `--checkpoint` 时优先加载 `best.pth.tar`；只有不存在 best checkpoint 时才回退到最新 epoch checkpoint。

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

### 1轮 smoke test

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic \
  --epochs 1 \
  --test_frequency 1

python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --epochs 1 \
  --test_frequency 1
```

## Test

测试时 `--degradation_mode` 必须与训练 checkpoint 的退化模式一致。

默认优先加载 `best.pth.tar`：

```bash
python comparison/EMR-Diff/Test.py \
  --dataset PaviaU \
  --degradation_mode gaussian_bicubic

python comparison/EMR-Diff/Test.py \
  --dataset PaviaU \
  --degradation_mode physical
```

指定 checkpoint：

```bash
python comparison/EMR-Diff/Test.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --checkpoint comparison/EMR-Diff/checkpoints/physical/PaviaU/model_epoch_100.pth.tar
```

## Experiment storage rule

两种退化模式的实验产物完全隔离。

### Checkpoints

```text
comparison/EMR-Diff/checkpoints/
├── gaussian_bicubic/
│   ├── PaviaU/
│   ├── Houston13/
│   └── Chikusei/
└── physical/
    ├── PaviaU/
    ├── Houston13/
    └── Chikusei/
```

### Logs

```text
comparison/EMR-Diff/logs/
├── gaussian_bicubic/<Dataset>/train_loss.csv
├── gaussian_bicubic/<Dataset>/validation_history.csv
└── physical/<Dataset>/...
```

### Outputs

```text
comparison/EMR-Diff/outputs/
├── gaussian_bicubic/<Dataset>/
│   ├── prediction_test_*.mat
│   ├── validation_metrics.txt
│   └── metrics.txt
└── physical/<Dataset>/...
```

`metrics.txt` 只用于最终测试；`validation_metrics.txt` 用于训练期验证。

## Important protocol note

在引入独立验证区之前，旧版 EMR-Diff 训练代码把中心 128x128 区域既作为训练期评估区域又作为最终测试区域。该旧流程会造成测试集参与 epoch 选择，因此旧版 100/200/300 epoch 评估结果只能用于调试与趋势观察，不应直接作为采用新版协议后的正式最终测试结果。

新版代码已经将验证区与中心测试区完全分离。正式实验建议从新版空间划分重新训练，并以 `best.pth.tar` 在最终测试区上只执行最终测试。

## Shared implementation

LR-HSI 不由 EMR-Diff 自己重新实现，而是统一调用仓库根目录：

```text
degradations/
data_loader.py
```

这样 EMR-Diff 与后续加入的其他对比方法使用完全相同的 `gaussian_bicubic` 和 `physical` 观测模型。

## Dependencies

EMR-Diff 依赖 PyTorch、OmegaConf、SciPy、tqdm、timm。公共退化模块由 PyTorch 实现，不再依赖 OpenCV 才能得到正式退化结果。

## Migration note

本目录由 UFGNet 仓库中的已适配 EMR-Diff 迁移而来。正式对比流程不再依赖原始 EMR-Diff 自带的固定 Harvard/Chikusei 数据读取、固定 x8 设置、固定 34 通道或旧 31+3 SRF 二进制文件，而是统一使用本仓库公共数据、SRF、退化与独立验证管线。
