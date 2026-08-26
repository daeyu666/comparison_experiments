# HSI Super-Resolution 项目模板

高光谱图像超分辨率（HSI-MSI Fusion）项目的通用代码模板。

提供可直接复用的 dataloader、损失函数、评估指标、SRF 工具和通用训练辅助函数，
不依赖任何具体的模型结构。

## 对比实验目录

所有对比方法统一放在：

```text
comparison/<Method>/
```

每个方法独立保存自身代码、配置和实验产物，不再把不同方法的权重和结果散放在仓库根目录。例如：

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
    ├── checkpoints/     # EMR-Diff 自身模型权重
    ├── outputs/         # EMR-Diff 自身重建结果和指标
    └── logs/            # EMR-Diff 自身训练日志
```

目录规则固定如下：

- 模型权重保存到 `comparison/<Method>/checkpoints/`；
- 测试结果、重建结果、指标文件保存到 `comparison/<Method>/outputs/`；
- 训练日志保存到 `comparison/<Method>/logs/`；
- 不同对比方法之间只共享仓库根目录的数据、评价指标及明确统一的实验协议代码；
- 新增方法继续在 `comparison/` 下新建独立方法文件夹，不再通过增加 Git 分支管理不同对比实验。

当前已迁入 `comparison/EMR-Diff/`。详细规则见 `comparison/README.md`，EMR-Diff 的运行说明见 `comparison/EMR-Diff/README.md`。

当前 EMR-Diff 对比协议为：

```text
scale factor = x4
LR-HSI = 5x5 Gaussian blur (sigma=2) + bicubic downsampling
HR-MSI = 8 uniformly selected bands
train patch = 64x64
stride = 32
test region = center 128x128
```

## 默认传感器协议

仓库根目录的通用模板默认采用真实光谱响应函数生成 HR-MSI：

- `PaviaU`：默认使用 **IKONOS 4-band**，即 Blue / Green / Red / NIR；标准 103-band PaviaU 使用 `data/wavelengths/PaviaU_nominal_430_860.txt` 与 `data/srf/ikonos_relative_spectral_response.csv`。
- `Houston13`、`Chikusei`：`auto` 模式下仍使用 WV2 all8。
- 原来的 **WV2 visible6** 没有删除，仍可通过 `--srf_band_set wv2_visible6` 显式启用，已有 `PaviaU_wv2_visible6_*` 权重文件也继续保留。

注意：上述内容是根目录通用模板默认设置；具体对比实验应以 `comparison/<Method>/README.md` 和该方法适配器中的统一对比协议为准。当前 EMR-Diff 明确覆盖为 8-band uniform MSI。

PaviaU 默认运行时无需额外指定传感器：

```bash
python main.py --dataset PaviaU
```

如需复现旧的 PaviaU WV2 六通道实验：

```bash
python main.py \
  --dataset PaviaU \
  --msi_mode srf \
  --srf_band_set wv2_visible6
```

## 模板内容

| 文件 | 说明 |
|------|------|
| `data_loader.py` | HSI 数据读取（.mat / h5）、预处理、patch 构建、DataLoader |
| `losses.py` | 光谱重建损失：SAM、光谱梯度、数据一致性、VQ commitment 等 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC 评估指标 |
| `srf_utils.py` | 光谱响应函数（SRF）加载、插值、权重构建、HSI→MSI 转换 |
| `prepare_srf_weights.py` | 预计算并保存 SRF 权重矩阵 |
| `utils.py` | 通用工具：随机种子、设备选择、checkpoint 存取、日志、CSV logger |
| `config.py` | 训练配置 dataclass + 命令行解析（不含模型参数） |
| `main.py` | 模板入口示例，展示如何串联各组件 |
| `analyze_spectral_regions.py` | 按光谱区域分析模型重建质量（模型通过参数传入） |
| `visualize_base_reconstruction.py` | 重建结果可视化：RGB 对比图、光谱曲线、误差图 |
| `comparison/` | 所有对比方法及其方法内独立实验产物目录 |

## 使用方式

1. 将原始数据放入 `data/raw/`。
2. 通用模板可在 `config.py` 的 `get_dataset_configs()` 中注册数据集。
3. 对比实验直接进入对应 `comparison/<Method>/` 运行。
4. 新增对比方法时，在 `comparison/` 下创建新方法目录，并保持权重、结果和日志方法内隔离。

```python
# 通用模板最小示例
from config import parse_args
from data_loader import build_loaders
from losses import SAMLoss, DataConsistencyLoss
from utils import set_seed, get_device

cfg = parse_args()
set_seed(cfg.seed)
train_loader, test_loader, info = build_loaders(cfg)

# model = YourModel(...)
# criterion = ...
# train loop ...
```

## 目录结构约定

```text
project/
├── data/                       # 公共数据和 SRF 权重
│   ├── raw/                    # 原始 HSI .mat 文件
│   ├── wavelengths/            # 各数据集波长文件
│   ├── srf/                    # 原始 SRF CSV
│   └── srf_weights/            # 预计算的 SRF 权重
├── comparison/                 # 对比实验
│   ├── README.md
│   └── <Method>/
│       ├── checkpoints/
│       ├── outputs/
│       └── logs/
├── models/                     # 通用模型定义（非独立对比实验）
└── ... shared utilities ...
```

## 扩展原则

- 根目录只保留与模型结构无关或多个实验共同使用的组件。
- 具体对比模型实现、训练逻辑、配置、权重、日志和结果全部放在 `comparison/<Method>/` 内。
- 对比实验统一在 `main` 分支管理，通过目录隔离方法，不通过大量分支隔离方法。
