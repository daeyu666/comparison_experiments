# Comparison Experiments

`comparison/` 专门用于存放所有对比方法。每个方法单独建立一个子目录，模型代码、权重、日志和实验结果均保持方法内隔离。

推荐结构：

```text
comparison/
├── README.md
├── EMR-Diff/
│   ├── ... source code ...
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
├── HSIFN/
│   ├── ... source code ...
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
├── PSRF-DiffNet/
│   ├── ... source code ...
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
└── <OtherMethod>/
    ├── ... source code ...
    ├── checkpoints/
    ├── logs/
    └── outputs/
```

## 目录规则

1. 每个对比方法使用 `comparison/<Method>/` 作为自己的工作根目录。
2. 模型权重只保存到该方法自己的 `checkpoints/`。
3. 训练日志与 loss 历史只保存到该方法自己的 `logs/`。
4. 重建结果、指标文件和中间实验输出只保存到该方法自己的 `outputs/`。
5. 不同方法只共享仓库根目录的数据、评价指标、SRF 和公共退化协议代码。
6. 新增方法优先保持原开源模型结构，只在方法适配层完成数据、通道、尺度、退化和训练控制接口适配。
7. 所有方法统一直接在 `main` 上维护，不为单个对比方法额外创建 Git 分支。

## 固定公平对比协议

所有正式对比实验统一使用 `x4` 超分，并且每个方法都必须支持常规退化与物理退化切换。

### LR-HSI 退化模式

```text
gaussian_bicubic:
  Gaussian PSF kernel = 5x5
  sigma = 2.0
  downsampling = bicubic x4

physical:
  MTF at LR Nyquist = 0.2
  Gaussian optical PSF derived from MTF
  detector pixel-area integration
  stride sampling x4
  PSF truncate = 3.0
```

正式结果必须复用仓库根目录公共退化算子。

### HR-MSI 传感器协议

| Dataset | MSI simulation | Channels |
|---|---|---:|
| PaviaU | IKONOS Blue / Green / Red / NIR SRF | 4 |
| Houston13 | WorldView-2 all8 SRF | 8 |
| Chikusei | WorldView-2 all8 SRF | 8 |

无论选择 `gaussian_bicubic` 还是 `physical`，上述 MSI 协议都不得改变。

### 非配准退化协议

公共非配准算子统一位于：

```text
degradations/misalignment.py
```

该实现与 `S2Diff/degradations/misalignment.py` 保持一致，支持：

```text
registered
translation
rotation
global
local smooth non-rigid
global + local
```

其中非配准只作用于 **HR-MSI**，HR-HSI GT 与 LR-HSI 保持不变。平移采用连续坐标与双线性采样，支持亚像素位移；正 `dx` 表示图像内容向右移动，正 `dy` 表示向下移动。局部非刚性形变由低分辨率控制网格生成并经 bicubic 插值得到连续位移场。

平移灵敏度实验默认采用：

```text
dx, dy ~ U(-d, d)
d = 0 / 0.5 / 1 / 2 / 3 / 4 / 6 px
```

同一 trial 在不同 `d` 下复用相同的归一化随机方向，以形成 paired sensitivity curve。

非配准实验的主指标采用 valid-overlap 口径：

```text
valid = valid_soft >= 0.999
PSNR_valid / SAM_valid
```

完整图像 PSNR / SAM / RMSE / ERGAS / SSIM / CC 作为辅助结果保留。

### Train / validation / test 空间划分

```text
train patch       = 64x64
train stride      = 32
validation region = fixed 128x128 region disjoint from test
final test region = center 128x128
```

训练 patch 必须同时避开验证区和最终测试区。训练过程中只允许访问训练集和验证集；最终测试区不得用于选择 epoch、调参或 early stopping。

### Early stopping 与验证频率

默认监控独立验证区 PSNR：

```text
monitor   = PSNR
min_delta = 0.02 dB
patience  = 2 validation evaluations
eval_seed = 1234
```

验证间隔不再固定为 100 epoch，而按数据集计算成本统一设置：

| Dataset | Validation interval |
|---|---:|
| PaviaU | 20 epochs |
| Houston13 | 10 epochs |
| Chikusei | 5 epochs |

这套验证频率属于同一数据集的统一公平协议。后续新增对比方法在可实现 early stopping 的情况下，应采用相同的数据集验证间隔，避免某个方法因为验证过稀而额外训练大量无效 epoch。

连续 2 次验证无有效提升则终止训练。对应最佳点之后的默认最大额外训练量约为 PaviaU 40 epoch、Houston13 20 epoch、Chikusei 10 epoch。

每次出现新最佳验证 PSNR，应额外保存 `best.pth.tar`；正式最终测试优先使用 `best.pth.tar`，而不是最后一个 epoch 的权重。

### 评价指标

正式对比统一调用仓库根目录 `metrics.py`：

```text
PSNR / RMSE / SAM / ERGAS / SSIM / CC
```

## 不同退化模式的实验产物隔离

```text
comparison/<Method>/checkpoints/<degradation_mode>/<Dataset>/
comparison/<Method>/logs/<degradation_mode>/<Dataset>/
comparison/<Method>/outputs/<degradation_mode>/<Dataset>/
```

## 当前方法

- `EMR-Diff/`：已接入公共 SRF、双退化、独立验证区、dataset-aware validation interval、best checkpoint 与 validation-based early stopping。
- `HSIFN/`：当前非配准正式对比方法。保留 HSIFN 两级 FlowNet、四尺度参考特征配准、置信掩码与 QRNN3D 融合主干；RGB 入口适配为 IKONOS/WV2 MSI，HR-MSI 对齐至固定 HSI/GT 坐标系；已接入公共 SRF、物理/常规退化、S2Diff 同口径非配准、valid-overlap 指标与 validation-based early stopping。
- `PSRF-DiffNet/`：保留此前复现实验代码用于追溯，但其官方参考坐标方向与当前“仅 HR-MSI 形变”的统一协议不匹配，不作为当前正式非配准对比结果。
