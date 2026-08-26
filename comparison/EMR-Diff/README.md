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

### 其余公共设置

```text
scale factor = x4
train patch = 64x64
stride = 32
test region = center 128x128
```

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
└── physical/<Dataset>/train_loss.csv
```

### Outputs

```text
comparison/EMR-Diff/outputs/
├── gaussian_bicubic/<Dataset>/
│   ├── prediction_*.mat
│   └── metrics.txt
└── physical/<Dataset>/
    ├── prediction_*.mat
    └── metrics.txt
```

`metrics.txt` 会记录数据集、退化模式、SRF profile 和统一评价指标，便于后续汇总常规退化与物理退化结果。

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

本目录由 UFGNet 仓库中的已适配 EMR-Diff 迁移而来。正式对比流程不再依赖原始 EMR-Diff 自带的固定 Harvard/Chikusei 数据读取、固定 x8 设置、固定 34 通道或旧 31+3 SRF 二进制文件，而是统一使用本仓库公共数据、SRF 与退化管线。
