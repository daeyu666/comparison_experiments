# HySure — Registered-only comparison

原方法：M. Simões, J. Bioucas-Dias, L. Almeida, J. Chanussot,
“A convex formulation for hyperspectral image superresolution via subspace-based
regularization,” IEEE TGRS 53(6), 3373–3388, 2015.

[官方实现](https://github.com/alfaiate/HySure) commit
`8652a5838b95c29ca056f7921f8ac58def6bbaf7`；
src/data_fusion.m blob `e12835582b1d309b9c0587ea44a3bd9523f57f44`。
参考 MATLAB 源码随目录保存在 reference_data_fusion.m（仅将注释编码转为 UTF-8）；
本 Python 移植遵循 GPL-3.0，许可见 LICENSE.upstream。

## 采用的算法版本

**HySure-SVD，已知公共 SRF/P0。**
以 LR-HSI 的前10个左奇异向量为光谱子空间，求解原方法的凸目标：

```text
min_X  0.5 ||Y_H - E P0(X)||_F^2
     + lambda_m/2 ||Y_M - R E X||_F^2
     + lambda_phi sum_pixel sqrt(sum_subspace(Dh(X)^2 + Dv(X)^2))
```

保留原 data_fusion.m 的四变量分裂 ADMM/SALSA、联合向量 TV 收缩和对偶更新。
默认200次、mu0.05；lambda_m1、lambda_phi5e-4、子空间维数10。
后两项正则权重及维数对应官方 Pavia HS/MS demo；
SVD 是原方法支持的子空间选择，官方 demo 默认选择的是 VCA。
本实现明确标识 SVD，不声称与 VCA demo 的有限迭代结果逐像素相同。

为匹配已有 benchmark，空间算子使用公共的**零边界高斯光学模糊+面积采样**，
而不是替换成原 demo 的周期模糊/点采样。
相应 X 正规方程使用共轭梯度求解（默认rtol1e-6、最多80步），
其余保真项、VTV正则和 ADMM 更新保持对应数学目标。
CG 未收敛会显式报错，不默默使用未满足容差的内层解。
SVD 仅由输入 LR-HSI 估计；直接使用公共已知 SRF/P0，
不运行作者 demo 的 blind 传感器估计、额外加噪或 GT 去噪。

## 数据与运行

六数据集沿用公共 loader、划分、归一化、SRF、x4 physical 观测和公共指标。
GT 仅生成观测和评分，求解器只接受 LR-HSI、HR-MSI、R、P0。
无需预训练或 checkpoint；每个配准测试 patch 只求解一次。
不执行 Mixed/Warp 测试。纯 NumPy/SciPy CPU float64，不依赖 MATLAB 或 Octave。

```bash
python -m pip install -r comparison/HySure/requirements.txt
python test.py --model HySure --dataset PaviaU --mode registered --device cpu

# 可选快速检查
python test.py --model HySure --dataset PaviaU --mode registered --device cpu --max_samples 1
```

可显式设置 --hysure_iterations、--hysure_rank、--hysure_lambda_phi、
--hysure_lambda_m、--hysure_cg_maxiter、--hysure_cg_rtol、
--classical_threads；参数全部记录，不用测试 GT 选择迭代或正则权重。
--classical_save_predictions 可另存 CHW 重建 NPY。

输出：outputs/registered/<Dataset>/summary.json、summary.csv、
per_sample_metrics.csv。包括目标函数值、子空间维数、完整求解参数、
SRF指纹、每对耗时和 Registered 指标，不输出 Warp/std。
--mode mixed 明确拒绝，train.py 会说明无需预训练。

## 验证

```bash
python -m pytest -q tests/test_traditional_registered.py
python comparison/traditional/smoke_test.py --full
```

CVXPY 只用于可选验证测试（不是运行依赖）：
测试将同一子空间/VTV问题显式构造成独立凸优化问题，
比较 ADMM 结果与 CLARABEL 参考最优值。
另验证公共 P0 的像素一致性与精确伴随、无 GT 求解、
Registered-only 路由及评分输出，并覆盖六组通道和128x128完整预算。

这是公共退化协议下的 HySure-SVD 适配；正式数据集指标需要实际运行后报告。

