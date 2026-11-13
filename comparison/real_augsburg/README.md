# Augsburg Region 2 · center-heldout Wald ×3

使用 [S2Diff-MH 的 center-holdout 缓存和标定](https://github.com/daeyu666/S2Diff-MH/blob/main/AUGSBURG2_WALD_CENTER_HOLDOUT.md)。
默认先从本仓库的 `data/augsburg2_wald_center_holdout` 查找缓存；若
`data/train/meta.json` 存在，也支持直接使用 `data` 作为缓存根目录。
本仓库没有缓存时，再查找 `../S2Diff-MH/data/augsburg2_wald_center_holdout`。
标定文件通常放在所用数据目录的 `calibration/Augsburg2_Wald_center_holdout_radiometry.json`。
路径不同时，训练和测试都可传 `--wald_root`（包含 `train/`、`validation/`、
`test/`、`full/` 的目录）和 `--radiometry_json`（标定文件路径）。

训练用 90m HSI + 从真实 Sentinel-2 聚合的 30m MSI 重建观测的 30m HSI。
训练块 24×24、步长 6，排除 30m 中心测试区及 6 像素保护带；验证用独立
`deep_valid`，48×48 块。默认 100 epoch、AdamW、学习率 1e-5、每 5 epoch
验证，按有效区 SAM 选 best。原尺度测试用观测 30m HSI + 真实 10m MSI，
仅输出留出的 144×144×242 中心 ROI；不添加模拟形变或使用 10m HSI 真值。

## 逐个训练

```bash
python train_real.py --model ESFS --device cuda:0
python train_real.py --model EMR-Diff --base_channels 128 --device cuda:0
python train_real.py --model EMR-Diff --base_channels 128 --device cuda:0 --resume auto
```

深度方法：MoG-DCN、PSRT、ESFS、CLSNet、EMR-Diff、M2DTN、PMI-RFCoNet、
MoE-PNP、SA-RFM、UAFL。CNMF、HySure 直接求解，不跑训练。
权重目录为 `comparison/<Method>/checkpoints/augsburg2_wald_center_holdout/`。
EMR-Diff 的默认宽度 64 保存 `best.pth.tar`，128 保存 `best_base128.pth.tar`，
两者日志及 last 权重独立；HSI+MSI 扩散状态都是 246 通道。
`--epochs` 允许先短跑；`--checkpoint_dir`、`--log_dir` 可隔离实验。

- CLSNet、EMR-Diff、UAFL 继续使用原有 Wald 模型、损失和检查点验证。
- MoG-DCN 的学习式空间算子使用 ×3 步长；ESFS、PSRT 按目标 MSI 尺寸工作。
- PMI-RFCoNet 保留三层配准/融合，在 LR、2×LR 和原生 HR 特征网格之间转换。
- M2DTN、MoE-PNP 保留原有 ×2→×2 主干，用原始 LR-HSI 和内部重采样的
  MSI，输出回到真实 ×3 网格后才做损失或评分。M2DTN 的小训练块会做配对边缘
  填充，评分前裁去。报告为 **Wald ×3 适配**。MoE-PNP 保留双尺度重建、
  NCC 和形变平滑损失；SA-RFM 保留 RFM + OTD。
- CNMF/HySure 使用共享 SRF 和 Wald 有效高斯 + ×3 面积积分，在 CPU 上对
  完整中心 ROI 求解；求解器的输入按原接口约束为非负，QNR 仍使用相同的观测
  和 train-only 标定。两者不估计几何形变。

## 自动找权重，输出 QNR

```bash
# 先预览：只看发现了哪些权重和将执行的命令
python test_real.py --model all --dry_run

# 只测已有权重的方法；CNMF/HySure 始终可直接求解
python test_real.py --model all --device cuda:0

# 权重分散在多个本地目录时，可以重复传入
python test_real.py --model all --device cuda:0 \
  --checkpoint_root /mnt/exp1/comparison_experiments \
  --checkpoint_root /mnt/exp2/comparison_experiments

# 只测一个宽度，或使用 --checkpoint 显式传入已验证的 Wald 权重
python test_real.py --model EMR-Diff --base_channels 128 --device cuda:0
```

按方法名和真实 Wald 目录查找 `best*.pth.tar`（也接受 .pth、.pt）。同目录同宽度
优先 best；只有传入 `--allow_last` 才在缺 best 时用 last。多个可用宽度会分别
评估。缺权重和运行失败分别记入总表，其他方法继续。
合成 `physical/PaviaU/best_base128.pth.tar` 等路径属于 ×4，不用于本实验；
即使重命名放入 Wald 目录，加载时也会核对协议及标定元数据。

默认输出 `outputs/augsburg2_wald_center_holdout/`，可用 `--output_dir` 改位置。
各方法/权重子目录统一写：

- `fused_hsi.npy`：H×W×242 float32；
- `fused_hsi.tif`：带中心 ROI 地理变换的 242 波段 GeoTIFF（需 rasterio，
  如只需 NPY，可加 `--no_geotiff`）；
- `qnr.json`：QNR、Dlambda、Ds 和评分窗口/有效区；
- `protocol.json`：权重、标定及推理来源。

总目录写 `summary.csv` 和 `summary.json`。默认神经方法使用 96×96 滑窗、
步长 48；正式比较统一使用同一滑窗设置。QNR 复用 S2Diff-MH / UAFL 的
HSI–MSI 实现：48×48 高分窗口、80% 最小有效像元比例、SRF 覆盖阈值 1%。
`--reference` 可额外运行独立的 30m Wald 参考测试，写到 `reference.log`，
不参与 10m 的 QNR 汇总。
