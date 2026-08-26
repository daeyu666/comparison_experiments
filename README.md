# HSI Super-Resolution Comparison Experiments

高光谱图像超分辨率（HSI-MSI Fusion）公共数据协议与对比实验仓库。所有正式对比方法统一放在 `comparison/` 下，并共享同一套数据、SRF、退化算子、空间划分和评价指标。

## 对比实验目录

```text
comparison/<Method>/
```

所有方法统一直接在 `main` 分支维护，不通过额外分支隔离不同对比实验。

## 所有对比实验固定协议

### 1. 超分尺度

```text
scale factor = x4
```

### 2. LR-HSI 双退化模式

#### 常规退化

```text
degradation_mode = gaussian_bicubic
Gaussian kernel = 5x5
sigma = 2.0
bicubic downsampling x4
```

#### 物理退化

```text
degradation_mode = physical
MTF at LR Nyquist = 0.2
MTF -> Gaussian optical PSF
detector pixel-area integration
stride sampling x4
PSF truncate = 3.0
```

所有新增对比方法都必须支持这两个模式，并复用仓库根目录公共实现。

### 3. HR-MSI 传感器协议

| Dataset | MSI simulation | Channels |
|---|---|---:|
| PaviaU | IKONOS Blue / Green / Red / NIR SRF | 4 |
| Houston13 | WorldView-2 all8 SRF | 8 |
| Chikusei | WorldView-2 all8 SRF | 8 |

### 4. Train / validation / test 空间协议

```text
train patch       = 64x64
train stride      = 32
validation region = fixed 128x128, disjoint from test
final test region = center 128x128
```

公共 `data_loader.py` 提供独立 train / validation / test loader。训练 patch 同时排除验证区和最终测试区；最终测试区只用于训练结束后对选定 best checkpoint 做正式评价。

### 5. Early stopping 与验证频率

默认早停规则：

```text
monitor = PSNR
min_delta = 0.02 dB
patience = 2 validation evaluations
eval_seed = 1234
```

验证间隔按数据集统一设置，不再固定每 100 epoch 才评估一次：

| Dataset | Validation interval | Max additional epochs after best under patience=2 |
|---|---:|---:|
| PaviaU | 20 | 40 |
| Houston13 | 10 | 20 |
| Chikusei | 5 | 10 |

这样 PaviaU 这种较小数据集不会在明显收敛后继续空跑很久，Chikusei 这种单 epoch 成本较高的数据集也能在接近收敛后快速触发验证和早停。

新最佳模型统一保存为：

```text
best.pth.tar
```

正式测试优先使用 `best.pth.tar`，而不是训练停止时最后一个 epoch 的权重。

### 6. 评价指标

```text
PSNR / RMSE / SAM / ERGAS / SSIM / CC
```

## 双退化模式实验产物隔离

```text
comparison/<Method>/checkpoints/<degradation_mode>/<Dataset>/
comparison/<Method>/logs/<degradation_mode>/<Dataset>/
comparison/<Method>/outputs/<degradation_mode>/<Dataset>/
```

## 公共组件

| 文件/目录 | 说明 |
|---|---|
| `config.py` | 公共数据、SRF 与退化模式配置 |
| `data_loader.py` | 公共 HSI 数据读取、train/validation/test 空间划分、patch 构建和观测生成 |
| `degradations/` | `gaussian_bicubic` 与 `physical` 公共退化算子 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 加载、插值、权重构建和 HSI→MSI |
| `comparison/` | 所有独立对比方法 |

## 扩展原则

- 新增方法放在 `comparison/<Method>/`。
- 所有方法必须支持 `gaussian_bicubic` 与 `physical` 两种 LR-HSI 退化。
- PaviaU 固定 IKONOS 4通道；Houston13 与 Chikusei 固定 WV2 8通道。
- 训练 patch 必须避开验证区与最终测试区。
- Early stopping 只能使用独立验证区，禁止使用最终测试指标选择 epoch。
- 同一数据集的对比方法尽量统一使用 PaviaU=20、Houston13=10、Chikusei=5 的验证间隔。
- 正式测试优先采用验证阶段选出的 best checkpoint。
