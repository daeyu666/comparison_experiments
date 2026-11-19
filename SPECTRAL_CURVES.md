# PaviaU / CAVE：非配准 Warp 逐波段绝对误差曲线

在 `comparison_experiments` 目录运行：

```bash
python plot_warp_spectral_curves.py
```

只读取先前 `export_paper_predictions.py` 保存的结果，不加载模型、不重新推理。
输入根目录默认 `outputs/paper_visualization`，严格读取：

```text
raw/<PaviaU或CAVE>/unregistered/warp/<方法>/<样本>/prediction.npy
```

默认展示 M2DTN、PMI-RFCoNet、MoE-PNP、SA-RFM、UAFL、Ours 的六条
逐波段绝对误差曲线：`abs(预测光谱 − GT光谱)`，越接近 0 越好。
全部方法使用不同颜色的实线；GT 作为误差基准，不另画一条 GT 曲线。
不混入 registered 输入，也不包含仅适用于配准数据的对比方法。
每个数据集只选一个像素，所有曲线都从同一参考坐标取值，不对任何方法
单独配准或改变取样位置。横轴使用导出时记录的实际中心波长（nm），
纵轴是归一化光谱强度的绝对差；不做曲线逐条归一化、平滑或手动缩放。
预测默认不裁剪，负值或超过 1 的值也如实参与误差计算。需要与裁剪指标口径对齐时，
可显式使用 `--clip_predictions`，该选择会记录在 JSON 中。
用 `--x_axis band` 可将横轴改为从 1 开始的波段编号。
用 `--plot_type spectrum` 可画原始强度曲线（包括 GT），也全部采用实线。

## GT 决定高光谱异质性像素

默认在每个 3×3 GT 邻域内，将每个像素的全波段光谱归一化为单位向量，
计算邻域所有光谱对之间的平均余弦差异（1−余弦相似度）。分数反映局部
光谱形状的差异，降低单纯亮度差异的影响。选点只依赖 GT，不查看任何
方法的重建误差或优势。

排除裁块边缘 8 像素、零光谱邻域，以及全波段均值低于
`max(0.01, 有效亮度第20百分位)` 的中心像素。在所有方法均已完整导出的
样本中，选择有效位置上分数最高的像素；并列时按样本名、行列顺序确定。
PaviaU 使用已导出的正式中心测试块；CAVE 默认遍历已导出的全部测试场景，
从中选出一个场景和一个像素，最终只有一张 CAVE 光谱图。
若输出目录已存在 `selected_pixels.json`，默认沿用原来的场景和像素，
让强度图与绝对误差图使用同一位置。用 `--reselect` 可重新按 GT 选点，
也可用 `--selection_json /path/to/selected_pixels.json` 指定原选点文件。
显式的场景/像素参数优先于保存的选点。

输出到 `outputs/paper_visualization/spectral_curves/`：

- `PaviaU_CAVE_warp_spectral_absolute_error.png/.pdf/.svg`：两个数据集并列的绝对误差图。
- `<数据集>/spectral_absolute_error.png/.pdf/.svg`：单数据集绝对误差图。
- `<数据集>/absolute_errors.csv`：波段编号、波长和六个方法的绝对误差。
- `<数据集>/spectra.csv`：实际波长、GT 与全部方法的逐波段数值。
- `<数据集>/selected_pixel_rgb.png`：GT RGB 中标出选点位置。
- `<数据集>/gt_heterogeneity.npy`：GT 异质性分数图。
- `selected_pixels.json`：场景、裁块内及原场景内的 0 起始行列坐标、选点规则、
  得分、候选场景得分、输入哈希、Warp 参数及权重来源。

使用 `--plot_type spectrum` 时，图片文件名仍为原来的 `spectral_curves`，
不覆盖新生成的绝对误差图。

可以先看选点位置，再指定场景或像素重画：

```bash
# 只在 jelly_beans 场景内选高异质性像素，PaviaU 仍自动选择
python plot_warp_spectral_curves.py --cave_sample jelly_beans

# 手动指定两个数据集在各自导出裁块中的 0 起始行列坐标
# 这里只是参数格式示例，请用 selected_pixels.json 中实际有效的坐标
python plot_warp_spectral_curves.py --paviau_pixel 60 70 \
  --cave_sample jelly_beans__r192_c192 --cave_pixel 50 60

# 修改输入/输出根目录
python plot_warp_spectral_curves.py --input_dir /path/to/paper_visualization \
  --output_dir /path/to/spectral_curves
```

需安装 NumPy、Matplotlib、Pillow。若没有一个样本同时包含全部六个方法
的 Warp 结果，脚本会列出各方法已有的样本数量并停止，不会静默省略曲线。
