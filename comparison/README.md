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

公共退化模式由仓库根目录 `degradations/` 与 `data_loader.py` 实现：

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

默认模式为：

```text
degradation_mode = gaussian_bicubic
```

所有新增对比方法必须提供等价的模式入口，例如：

```bash
--degradation_mode gaussian_bicubic
--degradation_mode physical
```

禁止某个方法自行实现另一套退化公式后仍标记为同名模式。正式结果必须复用仓库根目录公共退化算子。

### HR-MSI 传感器协议

MSI 模拟方式与 LR-HSI 退化模式相互独立，始终固定为：

| Dataset | MSI simulation | Channels |
|---|---|---:|
| PaviaU | IKONOS Blue / Green / Red / NIR SRF | 4 |
| Houston13 | WorldView-2 all8 SRF | 8 |
| Chikusei | WorldView-2 all8 SRF | 8 |

即：

```text
PaviaU    -> IKONOS SRF -> 4-channel MSI
Houston13 -> WV2 all8 SRF -> 8-channel MSI
Chikusei  -> WV2 all8 SRF -> 8-channel MSI
```

无论选择 `gaussian_bicubic` 还是 `physical`，上述 MSI 协议都不得改变。

### Train / validation / test 空间划分

为避免 early stopping 使用最终测试区造成测试泄漏，公共数据管线固定使用三部分空间划分：

```text
train patch      = 64x64
train stride     = 32
validation region = fixed 128x128 region disjoint from test
final test region = center 128x128
```

验证区优先取图像左上角 128x128；若与中心测试区发生重叠，则公共 `data_loader.py` 自动依次尝试其余角落，确保验证区与测试区不重叠。

训练 patch 必须同时避开验证区和最终测试区。训练过程中只允许访问训练集和验证集；最终测试区不得用于选择 epoch、调参或 early stopping。

### Early stopping

对需要迭代训练的对比方法，默认采用独立验证区 PSNR 进行早停：

```text
monitor            = PSNR
min_delta          = 0.02 dB
patience           = 2 validation evaluations
evaluation interval = method config, EMR-Diff default 100 epochs
eval_seed          = 1234
```

含义：只有当验证 PSNR 比历史最佳至少提高 0.02 dB 时才视为有效提升；连续 2 次验证无有效提升则终止训练。扩散/随机模型验证时必须固定评估随机种子，避免随机噪声造成假提升或误触发早停。

每次出现新最佳验证 PSNR，应额外保存：

```text
best.pth.tar
```

正式最终测试优先使用 `best.pth.tar`，而不是最后一个 epoch 的权重。

### 评价指标

正式对比统一调用仓库根目录 `metrics.py`：

```text
PSNR / RMSE / SAM / ERGAS / SSIM / CC
```

## 不同退化模式的实验产物隔离

同一方法、同一数据集在两种退化模式下得到的权重和结果必须分开保存：

```text
comparison/<Method>/checkpoints/<degradation_mode>/<Dataset>/
comparison/<Method>/logs/<degradation_mode>/<Dataset>/
comparison/<Method>/outputs/<degradation_mode>/<Dataset>/
```

例如：

```text
comparison/EMR-Diff/checkpoints/
├── gaussian_bicubic/
│   ├── PaviaU/
│   └── Chikusei/
└── physical/
    ├── PaviaU/
    └── Chikusei/
```

这样常规退化与物理退化不会互相覆盖 checkpoint、日志或测试结果。

## 当前方法

- `EMR-Diff/`：已接入公共 SRF、双退化、独立验证区、best checkpoint 与 validation-based early stopping。
