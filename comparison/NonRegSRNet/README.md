# NonRegSRNet reproduction

论文：**NonRegSRNet: A Nonrigid Registration Hyperspectral Super-Resolution Network**，
Ke Zheng, Lianru Gao, Danfeng Hong, Bing Zhang, Jocelyn Chanussot，
IEEE TGRS 60 (2022)，DOI 10.1109/TGRS.2021.3135501。

官方代码：[saber-zero/NonRegSRNet](https://github.com/saber-zero/NonRegSRNet)；
固定commit：`84e236db23695120ce0eb2a2f25b78d7fde70c21`。
活动网络来自`DeformatedHyperImageFusion/model/network.py`，
blob `6f23cef4a6371b6d99d58087b2df4e58097a6d73`。
损失与投影来自同目录`deformated_fusion.py`，配置来自官方`bash.sh`。

## 协议：保留逐观测对自监督优化

NonRegSRNet的官方方法对每对LR-HSI/HR-MSI优化网络，而后从MSI丰度和共享
光谱字典输出HR-HSI。LR-HSI通过自监督损失和字典优化影响输出。
本目录严格保留这一机制：**每次观测变化，都从相同随机种子重新拟合一套参数**。
GT和形变真值不传入优化器；固定最后一次迭代输出，不用GT挑选迭代、提前停止或筛选case。

普通监督方法的Registered→Mixed两阶段预训练不适用于该实现。
`--mode registered`表示配准观测测试；
`--mode mixed`表示共享非配准测试场景，不表示训练了10%/90%的混合checkpoint。
根`train.py --model NonRegSRNet`会明确解释这一差异并给出正确入口。

若冻结参数直接测试，官方重建头的HR-HSI只由HR-MSI计算，无法反映改变LR-HSI的影响；
本实现不会把这种固定权重的MSI-only运行报告为NonRegSRNet非配准结果。
与传统逐观测优化方法一样，需要在论文表中说明测试时自监督优化预算及运行时间。

## 网络与实际损失

- HSI与MSI丰度编码器：官方64基宽，1×1卷积通道128→256→512→num_rho；
- 默认num_rho=100，遵循官方发布demo；共享线性丰度→HSI解码器；
- 形变网络：官方UNet，保持代码中关闭pooling的行为，LR分辨率贯穿网络；
- STN：官方手写双线性插值，形变方向为HSI丰度到可靠MSI坐标；
- 保留非负位移输出、clamp丰度、字典/PSF投影；
- 默认盲退化估计：学习scale×scale、stride=scale的PSF及SRF支持区间内的响应权重。
  共享SRF曲线用于生成benchmark观测，模型使用其波段支持范围；不替换为已知PSF的监督网络。

损失与官方有效代码一致，像素项及丰度和为1约束使用sum reduction：

```text
L = ||Y_H - D(A_H)||_1
  + beta * sum|sum(A_H)-1|
  + alpha * ||Y_M - R(D(A_M))||_1
  + beta * sum|sum(A_M)-1|
  + delta * ||D(W(A_H)) - P(D(A_M))||_1
  + gamma * (1 - NCC(W(A_H), P(A_M)))
  + theta * smooth(flow)
```

alpha=1，beta=0.01，gamma=0.01，delta=1，theta=0.01，NCC窗口=1（官方mu参数）。
保留官方NCC的通道聚合、padding和归一化方式，没有擅自替换成另一种NCC。
Adam lr=3e-4、betas=(0.9,0.999)、无weight decay；位移网络lr为其0.5倍。
默认10000次/观测对，官方LambdaLR：前2000次恒定，后8000次衰减。

兼容性修正：按输入device分配采样/NCC张量，移除激活原地clamp对autograd的冲突，
为SRF投影的全零行提供均匀权重回退；普通非零情形与官方公式一致。
不依赖visdom、hues、dominate或过时skimage接口。

## 共享benchmark入口

从仓库根目录执行：

```bash
python -m pip install -r comparison/NonRegSRNet/requirements.txt

# 配准观测，所有heldout patches；无需先训练checkpoint
python test.py --model NonRegSRNet --dataset PaviaU --mode registered

# 正式主表：10个geometry seed × 每seed每patch 10个case，Registered评估一次
python test.py --model NonRegSRNet --dataset PaviaU --mode mixed

# 固定6档形变
python test_fixed_deformation.py --model NonRegSRNet --dataset PaviaU --mode mixed

# 直接调用主表入口，可控制单对优化预算
python test_multiseed_deformation.py --model NonRegSRNet --dataset PaviaU --nonreg_iterations 10000
```

6个数据集沿用根目录共享loader/SRF/划分：PaviaU、Houston13、Chikusei、
CAVE、Botswana、Augsburg synthetic。公共输入观测保持physical x4、
MTF@Nyquist=0.2、truncate=3；形变先作用于HR-HSI，再生成LR-HSI，
HR-MSI及评分GT保持可靠坐标。优化器看不到GT或生成的形变场。
最终指标与固定、多seed测试直接复用公共实现。

**计算预算为每一对观测10000次更新**，不能将单对10000次理解为整个数据集的训练总量。
完整多seed评估会执行大量独立拟合；先用以下smoke命令验证环境：

```bash
python comparison/NonRegSRNet/smoke_test.py --device cpu
python test.py --model NonRegSRNet --dataset PaviaU --mode mixed \
  --geometry_seeds 10001 --cases 1 --max_samples 1 --nonreg_iterations 2
```

该结果会标记smoke，不能用于正式论文。
固定预算可以通过`--nonreg_iterations`修改；相应lr、num_rho及损失系数均可使用
`--nonreg_<参数名>`显式设置，并写入输出元数据。
方法不参加当前真实Augsburg Wald入口；本次接入范围为共享synthetic x4。

## 缓存与断点

默认每次独立拟合，不保存所有观测对的大量网络权重。
需要中断续算时，显式指定同一个缓存目录：

```bash
python test.py --model NonRegSRNet --dataset PaviaU --mode mixed \
  --nonreg_cache_dir comparison/NonRegSRNet/checkpoints/pair_cache
```

每对观测的缓存键包含LR-HSI、HR-MSI、SRF、倍率、全部优化超参数、随机种子和结构版本的SHA256。
缓存每500次更新保存模型、Adam、scheduler及完成次数；只有同一观测、同一配置可恢复。
不同case独立初始化；缓存不会跨观测迁移字典或优化器状态。
公共多seed结果本身还支持`--resume_output`，行为与其他方法一致。

## 单独拟合自己的观测对

```bash
python comparison/NonRegSRNet/train.py \
  --pair observed_pair.npz --srf_weights srf_weights.npy \
  --layout hwc --scale_ratio 4 --device cuda:0 \
  --nonreg_iterations 10000 --output_dir comparison/NonRegSRNet/outputs/custom_pair
```

NPZ只需`lr_hsi`与`hr_msi`；SRF权重形状为MSI通道×HSI通道。
输出`fused_hsi.npy`和`fit_protocol.json`，不需要HR-HSI标签。

## 验证

```bash
python -m pytest -q tests/test_nonregsrnet.py
NONREG_TEST_OFFICIAL=1 python -m pytest -q tests/test_nonregsrnet.py
```

检查实际前向/反向、全部损失分支、保留LR尺度的UNet、两种观测独立拟合后的LR依赖、
no_grad评分环境内的优化、外部RNG恢复、缓存一致性、零SRF投影、共享入口。
可选源码对照从固定commit读取官方网络并校验git blob SHA，比较相同参数下的重建、
采样丰度与完整有效目标。CPU smoke覆盖6组传感器波段接口。
