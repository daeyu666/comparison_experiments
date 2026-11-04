# PMI-RFCoNet comparison adapter

论文：**Progressive Multi-Iteration Registration-Fusion Co-Optimization Network for
Unregistered Hyperspectral Image Super-Resolution**，IEEE TGRS 62 (2024)，
DOI [10.1109/TGRS.2024.3408424](https://doi.org/10.1109/TGRS.2024.3408424)。

官方实现：[Jiahuiqu/PMI-RFCoNet](https://github.com/Jiahuiqu/PMI-RFCoNet)。
固定源码 commit：`b52e0c84291b917574034da9081c3fac404d34e5`；
`model.py` blob：`53258b0de2089a788bde134e715114e3291a37e1`。
MIT 许可见 LICENSE.upstream。

## 是否适合当前比较

这是数据集级监督训练方法。训练后使用固定 checkpoint 对每个观测对执行一次前向；
论文名称中的 multi-iteration 指网络内部固定的三个层级，不是测试时反复训练。
因此共享的 10 geometry seeds × 10 cases 评测只需要重复前向，不需要每对优化 10000 次。

保留官方 LR/HR 编码器、三个 Co-RFB 层级、IRM、SRFU、位移范围 [-5,5]、
作者自定义双线性 STN 和输出 LeakyReLU。
保留发布代码注意力的实际张量运算，不替换成另一种注意力。
64×64 HR 训练 patch 对应 16×16 LR；最粗配准分支三次池化后为 2×2，
没有将特征压缩到 1×1。

## 原版配置与适配

| 设置 | 官方发布训练脚本 | 本适配器 |
|---|---|---|
| mid_channels | 64 | 64，所有数据集一致 |
| 三层级特征通道 | 256 / 128 / 64 | 原样保留 |
| num_fc / dropout | 4 / 0 | 4 / 0 |
| 损失 | mean L1(output, GT) | 相同 |
| 优化器 | Adam，lr=0.001，wd=0 | 相同 |
| 学习率调度 | MultiStepLR，[50,150,300]，gamma=0.1 | 每个训练阶段独立使用 |
| 批量 / HR patch | 4 / 160 | 公共协议 1 / 64 |
| 训练轮数 | 500 | 公共协议 Registered 800，Mixed 600 |
| HSI/MSI | 示例 Pavia Center 102/4 | 按共享数据与 SRF 确定 |

官方 test.py 和 model.py 的演示使用 mid_channels=128，与 train.py 的 64 不一致。
本实现以实际 train.py 为基准，保存结构元数据，测试时从 checkpoint 恢复结构，
不会用 128 通道测试一个 64 通道训练的模型。
源文件保留未参与 forward 的参数，以维持原版 state_dict 拓扑。

对官方核心仅做两类兼容适配：
1. 把硬编码的 4 个 MSI 输入通道改为传感器的实际通道数；4 通道时结构不变。
2. STN 的网格、索引与辅助张量使用输入 device，支持 CPU 和任意 CUDA 设备；
   保留作者插值公式、零边界处理及梯度。

没有已知 PSF/SRF 损失模块被替换：这是端到端监督模型，使用公共生成观测和 HR-HSI 标签训练。
本次复现范围为六个数据集的 synthetic x4；官方三层级拓扑固定 x4，
真实 Augsburg Wald x3 不接入。

## 公共实验设置

PaviaU、Houston13、Chikusei、CAVE、Botswana、Augsburg synthetic 共用根 loader、
划分、归一化和传感器 SRF。HR 训练 patch64、stride32；验证测试128；倍率4。
公共 LR 退化为 physical、MTF@Nyquist=0.2、truncate=3。

先完成 Registered 训练；Mixed 从 Registered best checkpoint 初始化。
Mixed 为 10% identity + 90% 非配准，形变先作用于 HR-HSI，再生成 LR-HSI；
MSI 和 GT 保持可靠坐标。平移±4 HR像素、旋转±2°、local≤4 HR像素、
grid5、minJac0.5。验证仅使用独立验证区域；Registered 监控 reg_PSNR，
Mixed 监控 warp_PSNR；测试指标完全复用公共 metrics。

## 运行

在仓库根目录，沿用已配置的 PyTorch 环境：

```bash
python -m pip install -r comparison/PMI-RFCoNet/requirements.txt
python comparison/PMI-RFCoNet/smoke_test.py --device cuda:0 --full

python train.py --model PMI-RFCoNet --dataset PaviaU --mode registered
python train.py --model PMI-RFCoNet --dataset PaviaU --mode mixed

python test.py --model PMI-RFCoNet --dataset PaviaU --mode registered
python test.py --model PMI-RFCoNet --dataset PaviaU --mode mixed
```

其他数据集替换 --dataset 即可。正式 Mixed 测试会输出 Registered 和 Warp 两类结果，
Warp 汇总 10 个几何种子的均值与标准差。
快速检查（不作为正式指标）：

```bash
python test.py --model PMI-RFCoNet --dataset PaviaU --mode mixed \
  --geometry_seeds 10001 --cases 1 --max_samples 1
```

恢复训练包含模型、Adam、scheduler、CPU/CUDA/NumPy/Python/几何 RNG 状态：

```bash
python train.py --model PMI-RFCoNet --dataset PaviaU --mode mixed --resume auto
```

单独诊断训练可直接调用 comparison/PMI-RFCoNet/train.py 或 train_hsi_deformed.py；
--max_train_batches 仅用于冒烟检查，参数会记录在 checkpoint 和 run_protocol.txt。
正式运行不传该选项。

## 输出

- checkpoints/physical/<Dataset>/{best,last}.pth.tar
- checkpoints/hsi_warp_final/<Dataset>/{best,last}.pth.tar
- logs/<family>/<Dataset>/validation_history.csv
- outputs/unified_two_stage/<Dataset>/：Registered 或 legacy 结果
- outputs/multiseed_deformation/<Dataset>/：summary.csv、summary.json、per_seed.csv 等

checkpoint 记录模型结构、上游版本、传感器指纹、训练阶段、优化器和学习率调度。
不兼容的阶段、通道或 SRF 会明确报错。

## 验证

```bash
python -m pytest -q tests/test_pmi_rfconet.py
PMI_TEST_OFFICIAL=1 python -m pytest -q tests/test_pmi_rfconet.py
```

检查真实前向/反向、最深2×2尺寸、三个配准融合层级的有效梯度、
双输入依赖、固定参数推理、原版完整网络相同权重下的输出与梯度、
Registered→Mixed实际训练、scheduler恢复后与连续训练一致，
以及 Registered、固定强度、多seed三个评分入口。
smoke_test.py 默认使用正式 mid_channels=64，覆盖六组传感器；
--full 额外测试128×128前向并输出本机计时。

上述检查验证代码接入和源码行为；尚未在真实数据集上完成全程训练，
不将随机张量或短训练分数视为论文复现精度。

