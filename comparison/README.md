# Comparison Experiments

`comparison/` 专门用于存放所有对比方法。后续新增对比实验时，每个方法单独建立一个子目录，不再把模型代码、权重、日志和实验结果散放在仓库根目录。

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
2. 训练产生的模型权重统一保存到该方法自己的 `comparison/<Method>/checkpoints/` 下，不保存到仓库根目录的公共 checkpoint 目录。
3. 训练日志、loss 历史和运行记录统一保存到该方法自己的 `comparison/<Method>/logs/` 下。
4. 测试结果、重建结果、指标文件和中间实验输出统一保存到该方法自己的 `comparison/<Method>/outputs/` 下。
5. 不同方法之间只共享仓库根目录的数据、评价指标和明确需要统一的实验协议代码；方法自身的模型文件、配置、日志、权重和结果保持隔离。
6. 新增对比方法时优先保持原开源方法的核心结构，在该方法子目录内完成数据与尺度适配，不为每个方法创建额外 Git 分支；当前统一直接在 `main` 上维护。
7. `checkpoints/`、`logs/` 和 `outputs/` 中的大体积实验产物默认不提交 Git；需要长期记录的最终指标可整理为轻量 CSV、JSON、TXT 或 Markdown。

## 固定公平对比协议

所有对比实验统一使用以下观测协议，不允许各方法自行更换 MSI 模拟方式：

```text
scale factor = x4
LR-HSI = 5x5 Gaussian blur (sigma=2) + bicubic downsampling

PaviaU:
  HR-MSI = IKONOS SRF
  channels = Blue / Green / Red / NIR
  MSI channels = 4

Houston13:
  HR-MSI = WorldView-2 SRF all8
  MSI channels = 8

Chikusei:
  HR-MSI = WorldView-2 SRF all8
  MSI channels = 8

train patch = 64x64
stride = 32
test region = center 128x128
```

正式数据由仓库根目录公共 `data_loader.py` 生成，评价指标调用根目录 `metrics.py`，避免每个方法各自重新实现一套数据与指标逻辑。

传感器规则固定为：

| Dataset | MSI simulation | Channels |
|---|---|---:|
| PaviaU | IKONOS SRF | 4 |
| Houston13 | WorldView-2 all8 SRF | 8 |
| Chikusei | WorldView-2 all8 SRF | 8 |

后续加入任何新的对比方法，都必须复用上述数据和传感器协议。若某个原始开源实现使用不同 MSI 通道数，应在该方法适配层中改为本仓库统一协议，而不是修改公共协议去适配单个方法。

## 当前方法

- `EMR-Diff/`：EMR-Diff 对比实验，已适配当前统一数据与 SRF 协议；模型权重、日志和实验结果均保存在该目录内部。
