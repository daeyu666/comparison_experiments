# EMR-Diff Comparison

本目录保存 EMR-Diff 对比实验代码及其统一实验协议适配。后续与 EMR-Diff 有关的训练代码、配置、模型权重、训练日志和实验结果均保持在本目录内，避免与其他对比方法混放。

## Shared comparison protocol

EMR-Diff 通过 `dataset_loader/ufg_adapter.py` 复用仓库根目录的公共数据接口。所有对比实验统一使用以下协议：

```text
scale factor = x4
LR-HSI = 5x5 Gaussian blur, sigma=2 + bicubic downsampling

PaviaU:
  HR-MSI = IKONOS SRF
  channels = Blue / Green / Red / NIR
  MSI channels = 4

Houston13:
  HR-MSI = WorldView-2 all8 SRF
  MSI channels = 8

Chikusei:
  HR-MSI = WorldView-2 all8 SRF
  MSI channels = 8

train patch = 64x64
stride = 32
test region = center 128x128
```

EMR-Diff 不再使用 uniform band selection。PaviaU 固定使用 IKONOS 4通道 SRF，Houston13 与 Chikusei 固定使用 WV2 all8 SRF，和本仓库其他对比方法保持一致。

当前动态状态通道数为：

```text
PaviaU:    103 HSI + 4 MSI = 107 channels
Chikusei:  128 HSI + 8 MSI = 136 channels
```

Houston13 的状态通道数同样按实际 HSI 波段数加 8 个 WV2 MSI 通道自动确定。

评价指标统一调用仓库根目录 `metrics.py`。原始 HSI 数据统一放在：

```text
data/raw/PaviaU.mat
data/raw/Houston13.mat
data/raw/Chikusei.mat
```

## Train

从仓库根目录运行：

```bash
python comparison/EMR-Diff/Train.py --dataset PaviaU
python comparison/EMR-Diff/Train.py --dataset Houston13
python comparison/EMR-Diff/Train.py --dataset Chikusei
```

建议正式训练前先分别做1轮快速检查：

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
  --epochs 1 \
  --test_frequency 1

python comparison/EMR-Diff/Train.py \
  --dataset Chikusei \
  --epochs 1 \
  --test_frequency 1
```

也可以进入方法目录运行：

```bash
cd comparison/EMR-Diff
python Train.py --dataset PaviaU
```

## Test

默认加载对应数据集最新 checkpoint：

```bash
python comparison/EMR-Diff/Test.py --dataset PaviaU
python comparison/EMR-Diff/Test.py --dataset Chikusei
```

指定 checkpoint：

```bash
python comparison/EMR-Diff/Test.py \
  --dataset PaviaU \
  --checkpoint comparison/EMR-Diff/checkpoints/PaviaU/model_epoch_100.pth.tar
```

## Experiment storage rule

每个数据集的模型权重只保存在本方法目录：

```text
comparison/EMR-Diff/checkpoints/
├── PaviaU/
├── Houston13/
└── Chikusei/
```

训练日志与 loss 历史只保存在：

```text
comparison/EMR-Diff/logs/
├── PaviaU/train_loss.csv
├── Houston13/train_loss.csv
└── Chikusei/train_loss.csv
```

测试重建结果与指标只保存在：

```text
comparison/EMR-Diff/outputs/
├── PaviaU/
│   ├── prediction_*.mat
│   └── metrics.txt
├── Houston13/
└── Chikusei/
```

`checkpoints/`、`logs/` 与 `outputs/` 均属于 EMR-Diff 自己的实验产物目录。大体积权重、重建结果和运行日志默认由本目录 `.gitignore` 忽略，只保留目录占位文件。

后续若增加其他对比方法，在 `comparison/<Method>/` 下建立独立目录，并采用相同的 `checkpoints/`、`logs/`、`outputs/` 隔离规则。统一规范见 `comparison/README.md`。

## Dependencies

EMR-Diff 代码依赖 PyTorch、OmegaConf、SciPy、tqdm、timm。为了严格使用当前 `Gaussian 5x5 + bicubic` 数据协议，建议安装 OpenCV；否则公共数据加载器的无 OpenCV 回退路径不应作为正式对比结果使用。

## Migration note

本目录由 UFGNet 仓库中的已适配 EMR-Diff 迁移而来。正式对比流程不再依赖原始 EMR-Diff 自带的固定 Harvard/Chikusei 数据读取、固定 x8 设置、固定 34 通道或旧 31+3 SRF 二进制文件，而是使用本仓库公共数据管线，并保持 EMR-Diff 的模型与扩散逻辑在本目录内自包含。
