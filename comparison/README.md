# Comparison Experiments

`comparison/` 专门用于存放所有对比方法。后续新增对比实验时，每个方法单独建立一个子目录，不再把模型代码、权重和实验结果散放在仓库根目录。

推荐结构：

```text
comparison/
├── README.md
├── EMR-Diff/
│   ├── ... source code ...
│   ├── checkpoints/
│   └── outputs/
└── <OtherMethod>/
    ├── ... source code ...
    ├── checkpoints/
    └── outputs/
```

## 目录规则

1. 每个对比方法使用 `comparison/<Method>/` 作为自己的工作根目录。
2. 训练产生的模型权重统一保存到该方法自己的 `comparison/<Method>/checkpoints/` 下，不保存到仓库根目录的公共 checkpoint 目录。
3. 测试结果、重建结果、指标文件和中间实验输出统一保存到该方法自己的 `comparison/<Method>/outputs/` 下。
4. 不同方法之间只共享仓库根目录的数据、评价指标和明确需要统一的实验协议代码；方法自身的模型文件、配置、日志、权重和结果保持隔离。
5. 新增对比方法时优先保持原开源方法的核心结构，在该方法子目录内完成数据与尺度适配，不为每个方法创建额外 Git 分支。
6. `checkpoints/` 和 `outputs/` 中生成的大体积权重与重建结果默认不提交 Git；需要长期记录的最终指标可整理为轻量 CSV、JSON 或 Markdown。

## 当前统一对比协议

当前 EMR-Diff 对比实验使用：

```text
scale factor = x4
LR-HSI = 5x5 Gaussian blur (sigma=2) + bicubic downsampling
HR-MSI = 8 bands
train patch = 64x64
stride = 32
test region = center 128x128
```

正式数据由仓库根目录公共 `data_loader.py` 生成，评价指标调用根目录 `metrics.py`，避免每个方法各自重新实现一套数据与指标逻辑。

## 当前方法

- `EMR-Diff/`：EMR-Diff 对比实验，已适配当前统一数据协议。
