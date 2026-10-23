# MoE-PNP comparison adapter

官方实现：[Jiahuiqu/MoE-PNP](https://github.com/Jiahuiqu/MoE-PNP)，上游版本固定为
\`c4bb4ca02d3583c929cff0096579b58bb3504e9f\`。论文 DOI：
[10.1109/TNNLS.2024.3412528](https://doi.org/10.1109/TNNLS.2024.3412528)。

本目录保留官方 MoE-PNP 的 registration-fusion consistency 主干：光谱响应映射、
两级配准与融合更新、形变采样、迭代去噪和原始重建/配准一致性损失。适配改动为
动态 HSI/MSI 通道数、运行时空间变换矩阵、设备无关的形变采样，以及统一的
\`model(lr_hsi, hr_msi) -> hr_hsi\` 接口。

## 统一实验协议

| 项目 | 设置 |
|---|---|
| 数据集 | PaviaU、Houston13、Chikusei、CAVE、Botswana、Augsburg synthetic |
| 超分倍率 | x4 |
| LR-HSI 退化 | shared physical degradation，MTF@Nyquist=0.2，PSF truncate=3.0 |
| HR-MSI | 共享传感器 SRF；按数据集使用 IKONOS、WV2、Nikon D700、EO-1 ALI 或 Sentinel-2 SRF |
| 训练块 / 步长 / 验证与测试块 | 64 / 32 / 128 |
| Stage 1 | registered，800 epochs，seed=1001 |
| Stage 2 | mixed，600 epochs，seed=10；10% identity + 90% HSI deformation |
| 形变 | dx、dy 独立 U(-4,+4) HR px；旋转 U(-2,+2)°；5x5 B-spline 控制网格；局部幅度上限4 px；min Jacobian=0.5 |
| 优化器 | AdamW，lr=1e-5，weight decay=5e-5 |
| 损失 | 官方双尺度 L1 重建 + 0.001 局部 NCC 配准一致性 + 0.01 形变场平滑 |
| 验证 | 继承 \`experiment_protocol.py\` 的 dataset-aware validation interval；Stage 1 监控 registered PSNR，Stage 2 监控 warp PSNR |
| 测试 | 共享六项指标；mixed 主表测试继承10 geometry seeds × 10 cases协议 |

Stage 2 仅加载 Stage 1 模型权重并重新初始化 AdamW，不继承 Stage 1 优化器状态。
形变施加于 GT-HSI 后再经公共退化算子形成 LR-HSI，HR-MSI 保持在参考坐标系。

## 训练与测试

从仓库根目录执行：

\`\`\`bash
python train.py --model MoE-PNP --dataset PaviaU --mode registered
python train.py --model MoE-PNP --dataset PaviaU --mode mixed

python test.py --model MoE-PNP --dataset PaviaU --mode registered
python test.py --model MoE-PNP --dataset PaviaU --mode mixed

python test_fixed_deformation.py --model MoE-PNP --dataset PaviaU --mode mixed
\`\`\`

建议先进行1 epoch smoke run：

\`\`\`bash
python comparison/MoE-PNP/train.py --dataset PaviaU --epochs 1 --validation_interval 1 --device cuda:0
python comparison/MoE-PNP/smoke_test.py
\`\`\`

产物分别保存在本目录的 \`checkpoints/physical/<Dataset>/\`、
\`checkpoints/hsi_warp_final/<Dataset>/\`、\`logs/\` 和 \`outputs/\`。

## 重要复现边界

官方代码固定了102个 HSI 波段、4个 MSI 波段和空间矩阵尺寸。本适配解除这些
输入通道和空间尺寸限制，以支持共享六数据集协议；网络主体和主要损失项保留，
但这属于统一协议下的适配复现，不是未经修改的官方代码，也不代表复现了论文表格数值。
原始两级结构只支持 x4；本目录暂不提供 Augsburg-2 真实世界 Wald x3 训练/推理适配器。

官方实现的矩阵式空间变换具有较高计算和显存开销，默认训练 batch=1。建议先运行
smoke test，再启动正式训练。

## 依赖

优先使用仓库根目录的实验环境。必要依赖包括 PyTorch、torchvision、NumPy、SciPy、
tqdm，以及根目录数据加载器所需的数据读取依赖。
