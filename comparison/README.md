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
├── UAFL/
│   ├── ... source code ...
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
├── HSIFN/
│   ├── ... source code ...
│   ├── checkpoints/
│   ├── logs/
│   └── outputs/
├── PRFCoAM/
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
| CAVE | Nikon D700 RGB SRF | 3 |
| Botswana | EO-1 ALI multispectral SRF | 8 |
| Augsburg | Sentinel-2A B2/B3/B4/B8 SRF V4.0 | 4 |

无论选择 `gaussian_bicubic` 还是 `physical`，上述 MSI 协议都不得改变。

### 正式 registered/mixed 非配准训练协议

根目录 `experiment_protocol.py` 是正式两阶段训练与主表测试的协议来源；EMR-Diff、CLSNet、DRTNet 及后续接入方法应遵循这一数据流。

```text
Stage 1 (registered):
  GT-HSI X -> physical degradation P0(X) -> LR-HSI Y_H
  HR-MSI Y_M = R0(X) remains in the reliable coordinate system
  target = X

Stage 2 (mixed):
  10%: Y_H = P0(X)                         [exact identity]
  90%: Y_H = P0(W_phi(X))                  [deformed HSI observation]
  HR-MSI Y_M = R0(X) remains unchanged
  target = X
```

统一几何范围为：`dx,dy ~ U(-4,+4)` 个 HR 像素、旋转角 `U(-2,+2)` 度、5x5 B-spline 控制网格、局部位移幅度 `U(0,4)` HR 像素，并要求局部 Jacobian 不低于 0.5。正式混合训练中的形变施加在 GT-HSI 上，再执行公共物理退化形成 LR-HSI；HR-MSI 作为可靠参考保持不变。

正式主表测试对混合训练权重使用 10 个独立 geometry seeds，每个 seed 对每个测试样本生成 10 个形变案例。报告注册结果及形变结果；形变结果按每个 seed 的全测试集均值计算 mean 和 sample std（ddof=1）。

### 独立 HR-MSI 形变敏感性诊断

`degradations/misalignment.py` 仍提供仅作用于 HR-MSI 的独立敏感性诊断，用于 `eval_misalignment.py` 这类配对平移/旋转/局部形变曲线，不替代上述正式 mixed 训练协议。

该诊断支持：

```text
registered
translation
rotation
global
local smooth non-rigid
global + local
```

平移严重度 `d` 定义为二维欧氏总位移上限：

```text
r ~ U(0, d)
theta ~ U(0, 2pi)
dx = r cos(theta)
dy = r sin(theta)
sqrt(dx^2 + dy^2) = r <= d

d = 0 / 0.5 / 1 / 2 / 3 / 4 / 6 px
```

同一 trial 在不同 `d` 下复用相同的归一化随机半径和方向，形成 paired sensitivity curve。该诊断的主指标采用 `valid_soft >= 0.999` 对应的 valid-overlap PSNR/SAM，完整图像 PSNR/SAM/RMSE/ERGAS/SSIM/CC 作为辅助结果。

### Train / validation / test 数据划分

正式方法统一读取仓库根目录 `data/splits/*.json`；不得在方法子目录内重新
定义另一套split。

| Dataset | Validation | Final test |
|---|---|---|
| PaviaU | top-left 128x128 | center 128x128 |
| Houston13 | top-left 128x128 | center 128x128 |
| Chikusei | centered 2304x2048中的rows 128:256，16个128x128 | rows 0:128，16个128x128 |
| CAVE | 20-scene development pool中的4个固定validation scenes | 12个固定完整512x512 scenes |
| Botswana | top-left 128x128 | center 128x128 |
| Augsburg synthetic x4 | MDAS official deep_valid region | MDAS official sub_area_1 |

训练patch默认为64x64、stride32，并与validation/test隔离。训练过程中只允许
访问train/validation；最终test不得用于选择epoch、调参或early stopping。

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
| CAVE | 5 epochs |
| Botswana | 20 epochs |
| Augsburg synthetic x4 | 10 epochs |

这套验证频率属于所有现有及后续新增方法的统一协议，适用于registered
和mixed两个synthetic训练阶段。根 `train.py` 从
`experiment_protocol.py::DATASET_VALIDATION_INTERVAL` 自动传入
`--validation_interval`；新方法的两个训练adapter必须接收并实际使用
这个参数，不能用内部固定周期覆盖，并在日志中记录实际验证周期。
是否启用自动早停不影响验证周期的继承。真实世界Augsburg Wald保持独立
每5epoch验证。根入口的800/600epoch预算及patience=999999保持原设置。

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
- `UAFL/`：按 CVPR 2026 UAFL 的 SVD 解混、CFDA、SCACA、SCMF 和 abundance residual 重建主线复现；将原 3-band RGB reference 入口动态适配为 IKONOS/WV2 MSI 通道，并以当前 torchvision 的等价 modulated deform-conv 后端替代旧 `mmcv_full` CUDA 执行层。正式两阶段对比遵循统一的 LR-HSI 形变协议；仅 HR-MSI 形变的敏感性曲线属于独立诊断。
- `HSIFN/`：保留此前复现实验用于追溯。其官方多级 FlowNet / QRNN3D 结构在共享 64x64 train patch 下会逐级压缩到极小空间尺度，registered sanity 长期停留在约 31 dB，不作为当前正式对比结果。
- `PRFCoAM/`：保留此前复现实验用于追溯。已完成 Torch-2.6/cu124 Mamba 后端兼容，但其官方配准方向为 LR-HSI -> HR-MSI，且共享协议下 registered 基础融合能力明显低于 S2Diff（physical 约 36.9 dB；gaussian-bicubic 最佳约 39.1 dB），不作为当前首选正式对比方法。
- `PSRF-DiffNet/`：已将 CRN、FRN、互补融合及扩散 x0 预测的活动执行路径对齐官方实现，并接入统一 registered/mixed 训练、checkpoint、主表多 seed 测试和六个 synthetic 数据集。架构对齐后需重新训练，旧 checkpoint 不兼容；当前尚未实现真实世界 Augsburg Wald adapter。
