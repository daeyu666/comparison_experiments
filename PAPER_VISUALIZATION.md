# PaviaU / CAVE 论文定性对比

在 `comparison_experiments` 根目录运行两个独立入口（绘图需 NumPy、Pillow、Matplotlib）：

```bash
python export_paper_predictions.py --device cuda:0 --resume
python render_paper_visualizations.py
```

默认输出到 `outputs/paper_visualization/`：

```text
raw/<dataset>/inputs/<sample>.npz
raw/<dataset>/registered/<method>/<sample>/
raw/<dataset>/unregistered/{registered,warp}/<method>/<sample>/
figures/<dataset>/registered/<method>/<sample>/
figures/<dataset>/unregistered/{registered,warp}/<method>/<sample>/
figures/<dataset>/references/<sample>/gt_rgb.png
```

`PaviaU` 的样本名为 `center`。CAVE 样本名包含场景和左上角坐标，
如 `jelly_beans__r192_c192`。每个原始结果目录保存 `prediction.npy`
（CHW、float32、未裁剪）、`residual_rmse.npy`（逐像素的全波段 RMSE）、
`sam_deg.npy`（SAM 角度，零谱向量处为 NaN）和 `metadata.json`。
误差图使用裁剪至 [0,1] 的预测和 GT；SAM 作图中的灰色为无定义像素。
输入 NPZ 是所有方法共用的，包含 GT、HR-MSI、两份 LR-HSI、SRF、
局部形变场、裁块坐标、随机种子、实际形变强度和输入哈希。
已有结果需要输入和权重哈希均一致才能用 `--resume` 跳过。

绘图生成 `prediction_rgb.png`、`residual_rmse.png`、`sam_deg.png`。
RGB 默认使用最接近 650/550/470 nm 的三个波段，拉伸区间由每个 GT 裁块
的第 1 和第 99 百分位确定；同裁块所有方法、两种条件共用该区间。
每个数据集内，各方法的同一种误差图共用色标，默认上限取全部已保存
有效像素的第 99 百分位（直方图近似）。PaviaU 的 RMSE 图把这个默认上限
乘以 1.25（最高为 1），并采用幂指数为 3 的色阶，使低误差更接近深蓝，
同时保留高误差区的色彩对比。这个变换只用于显示，不进行空间平滑，
实际 RMSE 范围保存在设置文件。用 `--residual_gamma 1` 可改回线性色阶。
PaviaU 的 SAM 图采用较轻的幂指数 1.25，角度范围保持不变；
用 `--sam_gamma 1` 可改回线性色阶。CAVE 默认保持线性色阶。
RMSE 和 SAM 均使用用户提供的参考色条中提取的配色，
调色板保存在 `paper_error_colormap.json`。每个数据集只需使用一条
`error_colorbar.png`（另有可缩放的 `error_colorbar.svg`），宽 24、高 512，
下端深蓝、上端深红，不带文字、数值、刻度或边框。旧版的两个
`*_colorbar.png` 文件保留为同一条色带的副本，便于替换已有排版。
这是一条“低误差→高误差”的配色图例；RMSE 和 SAM 的单位、范围和
幂指数各自独立，并在 `display_settings.json` 中完整保留，跨误差类型
的相同颜色不代表相同数值。可显式传入
`--residual_max` 和 `--sam_max` 固定上限；出图后又补入新方法时，请统一
重画或继续使用原上限。默认只用最近邻将显示图放大 3 倍，可用 `--zoom 1`
输出原始 128×128 像素。

## 裁块与形变

- PaviaU：复用正式中心 128×128 测试块。
- CAVE：默认在 12 个留出场景各取一个 128×128 几何中心裁块，坐标
  `[192:320,192:320]`。只用于定性展示，正式指标仍按每场景 16 个测试
  网格块计算。要让定性裁块也属于正式测试网格，选
  `--cave_crop central-grid`（中央四块中固定选左上，起点为 128,128），
  或 `--cave_crop all-grid` 导出全部 192 块。改变裁块策略时建议用新输出目录。
- Registered：`LR-HSI=P0(GT)`；Warp：先变形 GT，再用相同的物理 ×4
  算子退化。HR-MSI 和 GT 都保持原坐标。刚性固定为横纵各 +4 HR px、+2°；
  B 样条局部随机场的峰值固定为 4 HR px（向量 L2），局部最小 Jacobian
  ≥0.5。几何按样本和 `--warp_seed` 固定，跨方法完全共享。
  这是一组固定强度的定性示例，不是主表随机形变的均值。

## 权重与选择器

从两个仓库共同父目录的 `模型权重/` 查找：

- 配准：CNMF、HySure（无权重）、MoG-DCN、PSRT、ESFS、CLSNet、EMR-Diff；
  其余权重路径为 `<方法>-checkpoints/physical/<数据集>/best.pth.tar`。
- 非配准：M2DTN、PMI-RFCoNet、MoE-PNP、SA-RFM、UAFL。每个方法读取
  `<方法>-checkpoints/hsi_warp_final/<数据集>/best.pth.tar`，同一份权重
  分别推理 Registered、Warp。
- EMR-Diff 优先使用有的话 `best_base128`，再试 `best`，也支持
  `--emr_width 128` 或 `--checkpoint EMR-Diff:registered:CAVE=/path/to/weights`。
  各方法选定的权重见 `PLAN`；文件后缀也支持 `.pth`、`.pt`、`.ckpt`。
- `ours` 出现在两组。默认查找 `S2Diff-MH-checkpoints`（或 `ours-checkpoints`）。
  `physical/<数据集>/` 需要 Stage A 扩散和 Stage B HPGC-Refiner；
  `hsi_warp_final/<数据集>/` 需要 geometry、适配扩散和 HPGC-Refiner。
  可命名 `diffusion.pth`、`refiner.pth` 和 `geometry.pth`，也支持原始角色
  文件名唯一匹配。非配准组沿用同一组混合训练权重；Registered 绕过 CDRDI，
  Warp 执行完整三阶段。需要消歧时通过 `--ours_config` 传 JSON：

```json
{
  "PaviaU": {
    "registered": {
      "weights": {
        "diffusion": "S2Diff-MH-checkpoints/physical/PaviaU/raw_direct.pth",
        "refiner": "S2Diff-MH-checkpoints/physical/PaviaU/refiner.pth"
      },
      "options": {"baseline_checkpoint_type": "legacy"}
    },
    "unregistered": {
      "weights": {
        "geometry": "S2Diff-MH-checkpoints/hsi_warp_final/PaviaU/geometry.pth",
        "diffusion": "S2Diff-MH-checkpoints/hsi_warp_final/PaviaU/diffusion.pth",
        "refiner": "S2Diff-MH-checkpoints/hsi_warp_final/PaviaU/refiner.pth"
      }
    }
  }
}
```

相对权重路径基于 `--weights_root`。示例的 `legacy` 只用于旧版
Raw Direct；标准新权重用 `standard`。若 refiner 记录了模型配置，
默认读取其元数据；需要时在 `options` 指定模型宽度、步数等。

默认从任一仓库的 `data/raw/` 查找数据，也可传 `--data_root`。
用 `--dry_run` 预览方法权重和阶段；缺权重会列明，不阻塞其他方法。
运行日志在 `logs/<数据集>/<组>/<方法>.log`，汇总在 `export_summary.json`。
选择器示例：

```bash
python export_paper_predictions.py --datasets CAVE --cave_scenes jelly_beans oil_painting \
  --groups unregistered --methods SA-RFM ours --device cuda:0 --resume
python render_paper_visualizations.py --datasets CAVE --samples jelly_beans oil_painting
```

## 排查旧检查点 SRF

```bash
python export_paper_predictions.py --datasets PaviaU --srf_audit
```

此命令只读配准组 MoG-DCN 和非配准组 M2DTN、PMI-RFCoNet、SA-RFM
的归档检查点及传感器文件，
无需场景数据或 GPU。 `SRF_AUDIT` 报告同时列出训练时的 SRF 参数、存档
哈希和当前共享观测的哈希。 `exact` 是逐字节相同；`roundoff` 表示重现
了存档哈希，且每个 MSI 波段权重的 L1 差值均不超过 `1e-6`，可继续
使用共享输入。 `different_sensor` 或 `unverified` 会停止对应方法的导出。
若能在训练环境中导出原始 SRF 矩阵，可传入
`--training_srf M2DTN:PaviaU=/path/to/training_srf.npy` 重新核对。
传入的文件只用于验证，所有方法仍读取同一份 HR-MSI。

只重跑 CAVE 的 M2DTN 时，选择的是方法所属的 `unregistered` 组；
这会用同一份权重导出 `registered` 和 `warp` 两种输入条件：

```bash
python export_paper_predictions.py --datasets CAVE --groups unregistered \
  --methods M2DTN --device cuda:0 --resume
```

## 更新 Ours mixed 校正版可视化

使用同一套非配准 mixed 校正版权重分别推理 Registered、Warp。只选择 `unregistered` 组，
不加载 `physical` 下独立训练的配准权重，不检查其他方法权重。
Refiner 自动按 `terminal_correction` 文件名查找，规则与形变强度实验一致。
在原输出目录覆盖 Ours 时第一次不加 `--resume`，避免复用旧权重结果。

```bash
python export_paper_predictions.py \
  --datasets PaviaU CAVE --groups unregistered --methods ours \
  --diffusion_output terminal_correction --device cuda:0 \
  --output_dir outputs/paper_visualization

python render_paper_visualizations.py \
  --datasets PaviaU CAVE --groups unregistered --methods ours \
  --input_dir outputs/paper_visualization --reuse_display_settings
```

第二步仅更新 Ours 的 `prediction_rgb.png`、`residual_rmse.png`、`sam_deg.png`，
复用各数据集原 `figures/<数据集>/display_settings.json` 中的误差上限、gamma 和 RGB 波长，
保证与已经排版的其他方法保持相同显示尺度；不改写其他方法的图。
如果是首次可视化且没有上述设置文件，省略 `--reuse_display_settings`。
CAVE 默认仍是每个测试场景中心 128×128，可通过 `--cave_crop` 保持与之前一致。
输出位于 `figures/<数据集>/unregistered/{registered,warp}/ours/<样本>/`。
