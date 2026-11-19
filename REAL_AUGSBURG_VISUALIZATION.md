# 批量可视化已保存的 Augsburg Region-2 真实结果

在 `comparison_experiments` 根目录运行：

```bash
python visualize_real_augsburg.py
```

原来的脚本也新增了批量入口：

```bash
python comparison/EMR-Diff/visualize_augsburg2_wald_center_holdout.py --all_saved
```

两种入口完全相同，只读取已有结果，不加载模型，不进行推理或重新计算指标。

默认读取共同父目录的：

```text
Real_world推理结果/augsburg2_wald_center_holdout/<方法>/<变体>/fused_hsi.npy
```

包括 CNMF、HySure、MoG-DCN、PSRT、ESFS、CLSNet、EMR-Diff、M2DTN、
PMI-RFCoNet、MoE-PNP、SA-RFM、ours，共 12 个方法，不包含 UAFL。
EMR-Diff 严格使用 `base128`，其他方法严格使用 `default`，不自动混入
`base64`、`solver`、`last` 等变体。ours 目录也接受 `S2Diff-MH` 或 `S2Diff`；
方法目录匹配忽略大小写、连字符和下划线。若没有 `fused_hsi.npy`，可读取
变体目录里唯一的 `*_HSI.npy`，兼容原有方法专属文件名。

真实数据缓存优先查找 `comparison_experiments/data/augsburg2_wald_center_holdout`，
随后查找 `comparison_experiments/data` 和同级 S2Diff-MH 的数据缓存。
需要 `roi.json` 与 `full/hr_msi.npy`；可以用 `--wald_root` 指定其他路径。
重建结果尺寸必须与 ROI 中的 144×144×242 留出区一致，不能误用全区域结果。
支持 HWC、CHW 或单样本批维的 NPY 数据。

默认输出到：

```text
comparison_experiments/outputs/augsburg2_wald_center_holdout/visualization_all/
    Augsburg2_original_MSI_redbox.png
    Augsburg2_original_MSI_heldout_ROI.png
    Augsburg2_all_methods_RGB.png
    <方法>/<变体>/reconstruction_rgb.png
    visualization_summary.csv
    visualization_provenance.json
```

总览图只保留一份，不在各方法目录复制；红框按原始 ROI 坐标绘制。
同样的图在重复运行时不改写。使用原有脚本相同的 Sentinel-2 B4/B3/B2
总览配色及 HSI RGB 波段 `[43,28,10]`（从 0 开始）。12 个重建结果先共同
计算 1%–99% 拉伸范围，所有方法共用，避免各自调整亮度影响比较。
源推理结果文件不改写，所有新图片、汇总和来源记录均放在独立输出目录。

```bash
# 检查每个方法实际选中了哪个文件
python visualize_real_augsburg.py --dry_run

# 自定义归档、数据和输出路径
python visualize_real_augsburg.py \
  --results_root '/path/to/Real_world推理结果/augsburg2_wald_center_holdout' \
  --wald_root /path/to/data/augsburg2_wald_center_holdout \
  --output_dir /path/to/figures

# 若某个方法文件名/目录特殊，可显式覆盖
python visualize_real_augsburg.py --method ours /path/to/ours_reconstruction.npy
```

某个结果缺失、尺寸错误或包含非有限数时，汇总记录失败原因，其他有效结果
继续绘图。补齐结果后完整重画，以便全部方法重新使用同一个拉伸范围。
