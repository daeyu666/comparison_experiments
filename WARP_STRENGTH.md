# 非配准方法：形变强度实验

在 `comparison_experiments` 中运行：

```bash
python test_warp_strength.py --device cuda:0 --resume
```

默认测试六个模拟数据集的完整正式测试集，以及 M2DTN、PMI-RFCoNet、
MoE-PNP、SA-RFM、UAFL、ours。每个方法只读取一组混合训练权重，
五档测试不重新训练。共同父目录的归档路径沿用可视化脚本：
`模型权重/<方法>-checkpoints/hsi_warp_final/<数据集>/best.pth.tar`。
ours 沿用 `S2Diff-MH-checkpoints`（也接受 `ours-checkpoints`）及
`--ours_config` 的 dataset/unregistered 配置，详见 [权重配置](PAPER_VISUALIZATION.md)。

| 强度系数 | 平移最大幅度（HR 像素，每轴） | 旋转最大幅度 | 局部形变峰值上限（HR 像素） |
| --- | --- | --- | --- |
| 0 | 0 | 0° | 0 |
| 0.25 | 1 | 0.5° | 1 |
| 0.50 | 2 | 1° | 2 |
| 0.75 | 3 | 1.5° | 3 |
| 1.00 | 4 | 2° | 4 |

dx、dy 和角度在相应正负范围随机采样；局部形变为零均值 5×5 控制网格
生成的三次 B 样条场，幅度从 [0,上限] 提议，筛选局部最小 Jacobian ≥0.5
的形变。每个块、每个随机组先生成 1.0 档形变，再用相同基础形变缩放出
其余四档，保证各档都符合 Jacobian 约束。所有方法读取逐字节相同的观测。
随机数流固定在 CPU，模型随机数和 GPU 不影响输入。0 档直接读取原始
registered LR-HSI，避免零形变重采样产生插值误差。

HSI 观测由 `P0(W_phi(GT))` 生成，采用物理 ×4、MTF=0.2、PSF truncate=3；
GT 和 HR-MSI 保持参考坐标，模型只能访问两份观测。测试模型的训练时
几何上限始终保持 4/2/4。ours 在全部五档都运行完整几何估计、扩散及
Refiner，不依据真实强度绕过几何估计。

每档默认每块 10 组随机形变；CAVE 使用 12 场景×16 正式网格块，即
192 块，不使用定性图的中心裁块。先对每个随机组遍历全测试集取块均值，
再报告 10 组的均值和总体标准差。0 档重复的观测相同，但为随机推理模型
保留相同的 10 组重建随机种子。确定性模型在 0 档标准差应为 0。
全部方法共用 `fixed_deformation_protocol.calc_metrics`，与现有 S2 正式
测试口径一致：PSNR/RMSE/SAM 使用 [0,1] 裁剪，SSIM/ERGAS/CC 使用原始
预测，SSIM 为全局统计版本。

统一表格：`outputs/warp_strength/summary.csv`，一行对应一个数据集、方法、
强度，默认共 6×6×5=180 行。包含 PSNR、SAM、RMSE、ERGAS、SSIM、CC
及各自标准差、样本数、随机种子、权重路径及哈希、输入哈希、状态及错误。
失败行不填虚假指标，其余方法继续测试。日志位于 `logs/<数据集>/<方法>.log`，
每档 JSON 位于 `results/<数据集>/<方法>/`。

```bash
# 只看权重规划，不跑推理
python test_warp_strength.py --datasets PaviaU CAVE --dry_run

# 先运行两个数据集，全部非配准方法及 ours
python test_warp_strength.py --datasets PaviaU CAVE --device cuda:0 --resume

# 重试单个方法；汇总表保留未选择方法已经得到的行
python test_warp_strength.py --datasets PaviaU --methods SA-RFM --device cuda:0 --resume
```

`--resume` 按每个强度完成的 JSON 跳过，输入哈希、权重哈希、推理随机种子
或模型配置改变时拒绝复用。共享输入每次从原始数据生成，以核对数据是否变化；
每次只临时缓存一个数据集的观测，完成该数据集后自动释放缓存。
可用 `--data_root`、`--weights_root`、`--s2diff_root` 指定服务器路径，
`--checkpoint 方法:数据集=/path/to/weights` 覆盖单个比较方法的权重，
`--training_srf 方法:数据集=/path/to/matrix.npy` 核对旧版 SRF。
修改 cases/seed/eval_seed 时请使用新输出目录，避免混用不同实验设置。
