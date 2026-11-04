# CNMF — Registered-only comparison

原方法：N. Yokoya, T. Yairi, A. Iwasaki, “Coupled nonnegative matrix factorization
unmixing for hyperspectral and multispectral data fusion,” IEEE TGRS 50(2), 2012.

源代码来自[作者发布页](https://naotoyokoya.com/Download.html)的
[CNMF_Python.zip](https://naotoyokoya.com/assets/zip/CNMF_Python.zip)（2016版）。
Archive SHA256: `29acccea7c2109a9a7bd2ec9fd64f7aa9e0611fb073e043e7afe6e5f58eab55f`。
CNMF.py SHA256: `f6625e815f8d668e401b4ad783276ebe052231d8c6cfd2f2ef77db0adc795ca7`。
作者：Naoto Yokoya，GPL-3.0；许可见 LICENSE.upstream。

## 实现与协议

直接对每对配准 LR-HSI/HR-MSI 做耦合非负矩阵分解，无需数据集预训练。
保留作者 VCA、丰度初始化、乘法更新、加权和为1约束、双传感器耦合、
停止条件和发布代码的默认一轮外更新路径。
Python 3 适配包括 print、尺寸整数除法及零分母保护。
端元数按发布代码取 max(min(30,HSI bands),VD)，再限制在可用波段/像素数内；
VCA 私有固定随机种子默认1234，恢复调用者 NumPy RNG。

本比较采用**已知公共传感器模式**：
直接输入公共 SRF，跳过作者 demo 的 R/偏置估计；
将高分辨率丰度降采样替换成公共 physical P0。
这不是额外估计退化的 blind/cross-calibration demo。
默认内迭代上限200、外更新1、内停止阈值1e-8、其余更新沿用原发布代码。
发布 top-level 最后组合的是更新后的 HSI 字典与初始化阶段的 HR 丰度；
本实现与该实际输出路径一致，不暗中替换为另一种 CNMF 变体。

输入使用六数据集公共 loader、划分、归一化、SRF 和配准观测：
eval128、x4、MTF@Nyquist0.2、truncate3、零边界高斯PSF及面积采样。
GT 仅生成 benchmark 观测并评分，不进入求解器。
不训练 Registered/Mixed checkpoint，不执行任何 Warp/Mixed 测试。
程序运行在 CPU，使用 float64 数值求解和公共 float32 指标。

## 运行

仓库根目录，沿用已配置 PyTorch 环境：

```bash
python -m pip install -r comparison/CNMF/requirements.txt
python test.py --model CNMF --dataset PaviaU --mode registered --device cpu

# 可选：只检查一个测试 patch，并保存 CHW 重建数组
python test.py --model CNMF --dataset PaviaU --mode registered --device cpu \
  --max_samples 1 --classical_save_predictions
```

另外五个数据集替换 --dataset。
可显式设置 --cnmf_iterations、--cnmf_endmembers（0为上述自动规则）、
--classical_threads。默认参数不根据测试 GT 调整。
--mode mixed 会报出 Registered-only 提示，train.py 会说明无需预训练。

输出：outputs/registered/<Dataset>/summary.json、summary.csv、
per_sample_metrics.csv；可选 sample_XXXX.npy。
记录每个样本的迭代预算、实际端元数、源代码校验值、SRF指纹、耗时与指标。
Registered-only 只输出一类指标，没有 Warp/std 行。

## 验证

```bash
CLASSICAL_TEST_OFFICIAL=1 python -m pytest -q tests/test_traditional_registered.py
python comparison/traditional/smoke_test.py --full
```

源码对照会下载并验证作者 archive，在相同已知 SRF/P0 条件下比较完整融合输出。
公共空间算子与 Torch 实现对照，并检验伴随内积。
smoke 覆盖六组通道、128x128和默认完整预算。
本接入验证不代表已完成真实数据集的论文精度复现。

