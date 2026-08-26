# HSI Super-Resolution Comparison Experiments

高光谱图像超分辨率（HSI-MSI Fusion）公共数据协议与对比实验仓库。所有正式对比方法统一放在 `comparison/` 下，并共享同一套数据、SRF、退化算子和评价指标。

## 对比实验目录

```text
comparison/<Method>/
```

每个方法独立保存自身代码、配置和实验产物：

```text
comparison/
├── README.md
└── EMR-Diff/
    ├── Train.py
    ├── Test.py
    ├── arch/
    ├── model/
    ├── config/
    ├── dataset_loader/
    ├── checkpoints/
    ├── outputs/
    └── logs/
```

所有方法统一直接在 `main` 分支维护，不通过额外分支隔离不同对比实验。

## 所有对比实验固定协议

### 1. 超分尺度

```text
scale factor = x4
```

### 2. LR-HSI 退化必须支持两种模式切换

仓库根目录 `degradations/` 和 `data_loader.py` 提供所有对比方法共享的退化算子。

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

物理退化中的 Gaussian PSF 标准差由 `MTF_Nyq` 和尺度自动计算，随后执行光学模糊、探测器面积积分和空间采样。

**所有新增对比方法都必须支持这两个模式，并复用仓库根目录公共实现，不允许各方法自行定义另一套同名退化。**

推荐统一暴露：

```bash
--degradation_mode gaussian_bicubic
--degradation_mode physical
```

默认正式实验模式仍为：

```text
gaussian_bicubic
```

需要物理退化实验时显式切换为 `physical`。

### 3. HR-MSI 传感器协议固定

MSI 模拟协议与 LR-HSI 退化模式独立。无论使用常规退化还是物理退化，MSI 始终固定为：

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

PaviaU 标准 103-band 数据使用：

```text
data/wavelengths/PaviaU_nominal_430_860.txt
data/srf/ikonos_relative_spectral_response.csv
```

Houston13 与 Chikusei 使用各自波长文件及：

```text
data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv
```

### 4. Patch 与测试区域

```text
train patch = 64x64
stride = 32
test region = center 128x128
```

### 5. 评价指标

正式对比统一调用根目录 `metrics.py`：

```text
PSNR / RMSE / SAM / ERGAS / SSIM / CC
```

## 双退化模式实验产物隔离

每个方法都应按退化模式隔离训练产物：

```text
comparison/<Method>/checkpoints/<degradation_mode>/<Dataset>/
comparison/<Method>/logs/<degradation_mode>/<Dataset>/
comparison/<Method>/outputs/<degradation_mode>/<Dataset>/
```

避免同一数据集的常规退化与物理退化 checkpoint、日志和测试结果互相覆盖。

## 公共组件

| 文件/目录 | 说明 |
|---|---|
| `config.py` | 公共数据、SRF 与退化模式配置 |
| `data_loader.py` | 公共 HSI 数据读取、patch 构建和观测生成 |
| `degradations/` | `gaussian_bicubic` 与 `physical` 公共退化算子 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 加载、插值、权重构建和 HSI→MSI |
| `data/srf/` | IKONOS / WV2 SRF |
| `data/wavelengths/` | 数据集波长网格 |
| `comparison/` | 所有独立对比方法 |

## 扩展原则

- 新增方法放在 `comparison/<Method>/`。
- 模型权重、日志和结果全部保存在方法自己的目录内。
- 对比方法必须复用公共 `data_loader.py` 或对其做轻量适配，不能改变公平对比协议。
- 所有方法必须支持 `gaussian_bicubic` 与 `physical` 两种 LR-HSI 退化。
- PaviaU 固定 IKONOS 4通道；Houston13 与 Chikusei 固定 WV2 8通道。
- 两种退化模式下 MSI 传感器协议保持完全一致。
