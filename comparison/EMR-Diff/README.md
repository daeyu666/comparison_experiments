# EMR-Diff Comparison

本目录保存 EMR-Diff 对比实验代码及其公共实验协议适配。后续与 EMR-Diff 有关的训练代码、配置、模型权重和实验结果均保持在本目录内，避免与其他对比方法混放。

## Shared comparison protocol

EMR-Diff 通过 `dataset_loader/ufg_adapter.py` 复用仓库根目录的公共数据接口。当前固定协议为：

```text
scale factor = x4
LR-HSI = 5x5 Gaussian blur, sigma=2 + bicubic downsampling
HR-MSI = 8 uniformly selected bands
train patch = 64x64
stride = 32
test region = center 128x128
```

评价指标统一调用仓库根目录 `metrics.py`。

原始 HSI 数据统一放在仓库根目录：

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

快速检查：

```bash
python comparison/EMR-Diff/Train.py \
  --dataset PaviaU \
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
  --checkpoint comparison/EMR-Diff/checkpoints/EMRDIFF_PaviaU/model_epoch_100.pth.tar
```

## Experiment storage rule

模型权重只保存在本方法目录：

```text
comparison/EMR-Diff/checkpoints/
└── EMRDIFF_<dataset>/
    └── model_epoch_<N>.pth.tar
```

测试结果、重建结果、指标文件及其他实验输出只保存在：

```text
comparison/EMR-Diff/outputs/
└── <dataset>/
```

`checkpoints/` 与 `outputs/` 仅作为本地实验产物目录，大体积权重和重建结果默认由 `.gitignore` 忽略。

后续若增加其他对比方法，在 `comparison/<Method>/` 下建立独立目录，并采用相同的 `checkpoints/`、`outputs/` 隔离规则。统一规范见 `comparison/README.md`。

## Migration note

本目录由 UFGNet 仓库中的已适配 EMR-Diff 迁移而来。正式对比流程不再依赖原始 EMR-Diff 自带的固定 Harvard/Chikusei 数据读取和旧 SRF 二进制文件，而是使用本仓库公共数据管线，从而避免跨仓库路径依赖。
