# HSI Super-Resolution Comparison Experiments


## 统一训练 / 测试入口（正式两阶段对比实验）

从本协议起，新增和重跑的对比模型不再要求用户记忆每个
`comparison/<Method>/` 子目录下的训练命令。仓库根目录提供统一入口：

```bash
python train.py --model <Method> --dataset <Dataset> --mode registered
python train.py --model <Method> --dataset <Dataset> --mode mixed

python test.py --model <Method> --dataset <Dataset> --mode registered
python test.py --model <Method> --dataset <Dataset> --mode mixed
```

当前已经接入：

```text
UAFL
EMR-Diff
```

模型名大小写不敏感，并支持 `EMR` / `EMRDiff` 等常用别名。
数据集统一支持：

```text
PaviaU / Houston13 / Chikusei / CAVE / Botswana / Augsburg
```

除 `--device` 这一运行环境选项外，正式实验超参数不再从根入口暴露，
避免不同方法因手工命令不同而产生协议漂移。冻结设置集中定义在
`experiment_protocol.py`。新对比方法接入时只需要增加一次模型适配注册，
之后训练和测试仍使用上述根命令。

### 两阶段冻结训练协议

#### Stage 1：registered

```text
epochs = 800
LR-HSI = P0(X)
HR-MSI = R0(X)
GT-HSI = X

image_size = 128
train patch = 64
stride = 32
scale = x4

degradation = physical
MTF@Nyquist = 0.2
PSF truncate = 3.0

optimizer = AdamW
lr = 1e-5
weight decay = 5e-5
batch size = 1
```

Stage 1固定训练满800 epoch。正式根入口通过极大的early-stop patience避免
原方法自己的默认早停提前终止。best checkpoint仍由独立validation选择。

#### Stage 2：mixed

Stage 2从同一模型、同一数据集Stage 1的 `best.pth.tar` 初始化。
只加载**模型权重**，AdamW重新初始化，训练600 epoch。

每个训练样本按以下固定概率生成LR-HSI观测：

```text
10%:
  identity
  dx = 0
  dy = 0
  rotation = 0
  local = 0
  LR-HSI = P0(X)

90%:
  deformed
  dx, dy ~ U(-4, 4) HR pixels, independently
  rotation ~ U(-2, 2) degrees
  local amplitude proposal ~ U(0, 4) HR pixels
  5x5 control grid
  cubic B-spline dense local field
  min Jacobian >= 0.5
  LR-HSI = P0(W_phi(X))
```

在两种分支中：

```text
GT-HSI = X
HR-MSI = R0(X)
```

即只改变HSI观测几何，HR-MSI始终保持可靠坐标系。local amplitude的
`U(0,4)` 是采样proposal；为满足 `min Jacobian >= 0.5`，
最终接受样本是该proposal在非折叠约束下的条件分布。

Stage 2其余固定设置：

```text
epochs = 600
optimizer = AdamW
lr = 1e-5
weight decay = 5e-5
batch size = 1

validation:
  Registered + Warp
  warped validation cases = 5
  best checkpoint monitor = Warp PSNR

final test:
  registered checkpoint -> Registered
  mixed checkpoint -> Registered + Warp
  warped test cases = 10
```

Warp validation/test使用相同的几何范围：
`dx,dy~U(-4,4)`、`rotation~U(-2,2)`、
`local amplitude proposal~U(0,4)`、`min Jacobian>=0.5`。

### 最简运行方式

以Chikusei为例：

```bash
# UAFL
python train.py --model UAFL --dataset Chikusei --mode registered
python train.py --model UAFL --dataset Chikusei --mode mixed
python test.py  --model UAFL --dataset Chikusei --mode registered
python test.py  --model UAFL --dataset Chikusei --mode mixed

# EMR-Diff
python train.py --model EMR-Diff --dataset Chikusei --mode registered
python train.py --model EMR-Diff --dataset Chikusei --mode mixed
python test.py  --model EMR-Diff --dataset Chikusei --mode registered
python test.py  --model EMR-Diff --dataset Chikusei --mode mixed
```

换数据集时只修改 `--dataset`；换模型只修改 `--model`；
切换配准/混合训练只修改 `--mode`。

统一入口仍将checkpoint保存在各自方法目录：

```text
Stage 1:
comparison/<Method>/checkpoints/physical/<Dataset>/best.pth.tar

Stage 2:
comparison/<Method>/checkpoints/hsi_warp_final/<Dataset>/best.pth.tar
```

日志与输出也继续保留在各模型自己的 `logs/` 和 `outputs/` 下，
不会把不同模型产物混到仓库根目录。

### UAFL历史结果例外

**现有UAFL mixed checkpoint是在本次10%/90%统一协议确定之前训练完成的。**
旧版 `comparison/UAFL/train_hsi_deformed.py` 实际为deformed-only
stage 2，没有显式的10% identity Bernoulli分支。由于实验时间限制，
这些已经训练完成的UAFL权重本轮不要求重新训练。

必须保持以下标注：

```text
legacy UAFL mixed checkpoint:
  historical exception
  stage-2 training != formal 10% identity / 90% deformed protocol
```

测试脚本会在检测到旧checkpoint缺少 `registered_probability` 字段时
打印警告，且不会把该权重重新标记成10/90训练结果。

**从本次提交之后新训练或重跑的UAFL以及所有后续对比模型，一律使用
根目录统一入口和10% identity / 90% deformed协议。**


高光谱图像超分辨率（HSI-MSI Fusion）公共数据协议与对比实验仓库。所有正式对比方法统一放在 `comparison/` 下，并共享同一套数据、SRF、退化算子、空间划分和评价指标。

## 对比实验目录

```text
comparison/<Method>/
```

所有方法统一直接在 `main` 分支维护，不通过额外分支隔离不同对比实验。

## 所有对比实验固定协议

### 1. 超分尺度

```text
scale factor = x4
```

### 2. LR-HSI 双退化模式

#### 常规退化

```text
degradation_mode = gaussian_bicubic
Gaussian kernel = 5x5
sigma = 2.0
bicubic downsampling x4
```

#### 物理退化

```text
degradation_mode = physical
MTF at LR Nyquist = 0.2
MTF -> Gaussian optical PSF
detector pixel-area integration
stride sampling x4
PSF truncate = 3.0
```

所有新增对比方法都必须支持这两个模式，并复用仓库根目录公共实现。

### 3. HR-MSI 传感器协议

| Dataset | MSI simulation | Channels |
|---|---|---:|
| PaviaU | IKONOS Blue / Green / Red / NIR SRF | 4 |
| Houston13 | WorldView-2 all8 SRF | 8 |
| Chikusei | WorldView-2 all8 SRF | 8 |
| CAVE | Nikon D700 RGB SRF | 3 |
| Botswana | EO-1 ALI multispectral SRF (MS-1p/PAN excluded) | 8 |
| Augsburg | Sentinel-2A B2/B3/B4/B8 SRF V4.0 | 4 |

SRF曲线和波长资源与 `S2Diff-MH` 字节级同步，来源记录在
`data/srf/SOURCES.md`。Augsburg 242个EnMAP波长直接读取MDAS的
`band_242_meta_info.hdr`。

### 4. Train / validation / test 数据协议

所有正式对比方法必须与 `S2Diff-MH/data/splits/*.json` 使用完全相同的
split。不同数据集不再强行套用同一个center-128模板：

| Dataset | Train | Validation | Final test |
|---|---|---|---|
| PaviaU | 64x64 / stride32，排除val/test | top-left 128x128 | center 128x128 |
| Houston13 | 64x64 / stride32，排除val/test | top-left 128x128 | center 128x128 |
| Chikusei | center-crop 2304x2048；rows 256:2304，64x64/stride32 | rows 128:256，16个128x128 | rows 0:128，16个128x128 |
| CAVE | 固定16 scenes，64x64/stride32 | 固定4个完整512x512 scenes | 固定12个完整512x512 scenes |
| Botswana | 64x64 / stride32，排除val/test | top-left 128x128 | center 128x128 |
| Augsburg synthetic x4 | MDAS官方deep_train区域，64x64/stride32 | MDAS官方deep_valid区域 | MDAS官方sub_area_1区域 |

公共 `data_loader.py` 已实现以上协议并提供独立
`build_train_val_test_loaders()`。最终test严禁参与best epoch选择、调参或
early stopping。机器可读协议位于 `data/splits/`，两仓库内容保持同步。

### 5. Early stopping 与验证频率

默认早停规则：

```text
monitor = PSNR
min_delta = 0.02 dB
patience = 2 validation evaluations
eval_seed = 1234
```

验证间隔按数据集统一设置，不再固定每 100 epoch 才评估一次：

| Dataset | Validation interval | Max additional epochs after best under patience=2 |
|---|---:|---:|
| PaviaU | 20 | 40 |
| Houston13 | 10 | 20 |
| Chikusei | 5 | 10 |

这样 PaviaU 这种较小数据集不会在明显收敛后继续空跑很久，Chikusei 这种单 epoch 成本较高的数据集也能在接近收敛后快速触发验证和早停。

新最佳模型统一保存为：

```text
best.pth.tar
```

正式测试优先使用 `best.pth.tar`，而不是训练停止时最后一个 epoch 的权重。

### 6. 评价指标

```text
PSNR / RMSE / SAM / ERGAS / SSIM / CC
```

## 双退化模式实验产物隔离

```text
comparison/<Method>/checkpoints/<degradation_mode>/<Dataset>/
comparison/<Method>/logs/<degradation_mode>/<Dataset>/
comparison/<Method>/outputs/<degradation_mode>/<Dataset>/
```

## 公共组件

| 文件/目录 | 说明 |
|---|---|
| `config.py` | 公共数据、SRF 与退化模式配置 |
| `data_loader.py` | 公共 HSI 数据读取、train/validation/test 空间划分、patch 构建和观测生成 |
| `degradations/` | `gaussian_bicubic` 与 `physical` 公共退化算子 |
| `metrics.py` | PSNR / RMSE / SAM / ERGAS / SSIM / CC |
| `srf_utils.py` | SRF 加载、插值、权重构建和 HSI→MSI |
| `comparison/` | 所有独立对比方法 |

## 扩展原则

- 新增方法放在 `comparison/<Method>/`。
- 所有方法必须支持 `gaussian_bicubic` 与 `physical` 两种 LR-HSI 退化。
- PaviaU固定IKONOS 4通道；Houston13/Chikusei固定WV2 8通道；CAVE固定Nikon D700 3通道；Botswana固定EO-1 ALI 8通道；Augsburg固定S2A B2/B3/B4/B8 4通道。
- 训练 patch 必须避开验证区与最终测试区。
- Early stopping 只能使用独立验证区，禁止使用最终测试指标选择 epoch。
- 同一数据集的对比方法尽量统一使用 PaviaU=20、Houston13=10、Chikusei=5 的验证间隔。
- 正式测试优先采用验证阶段选出的 best checkpoint。


## 新增数据集下载地址与本地目录

- CAVE官方Columbia数据库：
  https://cave.cs.columbia.edu/repository/Multispectral
- Botswana Hyperion（UPV/EHU公开MATLAB数据）：
  http://www.ehu.eus/ccwintco/uploads/7/72/Botswana.mat
- Augsburg MDAS官方数据DOI：
  https://doi.org/10.14459/2022mp1657312
  （landing page: https://mediatum.ub.tum.de/1657312）

推荐本地结构：

```text
data/raw/Botswana.mat
data/raw/CAVE/<32 scene folders>/
data/raw/Augsburg/Augsburg_data_4_publication/
```

CAVE官方16-bit PNG读取需要`Pillow`；Augsburg多波段TIFF读取需要`tifffile`。
原始数据不提交到仓库，SRF/波长/划分manifest已提交并冻结。

## Verified data protocol (2026-09-27)

The audit is executable: `python check_data_protocol.py --data-root /path/to/data/raw --output audit.json`.
Run it in each repository with the same raw data root; add `--compare /path/to/S2Diff-MH-audit.json`
on the comparison run. Install the repository requirements plus `rasterio` for geographic checks.
It reads every CAVE scene, all nine official Augsburg SR TIFFs, all split coordinates,
and every validation/test sample. It checks SRF weights, finite values, shapes, geographic/scene
separation, and compares resource hashes, coordinates and sampled returned tensors across repositories.
Training samples overlap within the training split by design; cross-split overlap is zero.

| Dataset | Raw H x W x bands | Prepared H x W x bands | Train / validation / test samples | MSI channels |
|---|---|---|---|---|
| PaviaU | 610 x 340 x 103 | 608 x 340 x 103 | 110 / 1 / 1 | 4 |
| Houston13 | 349 x 1905 x 144 | 348 x 1904 x 144 | 470 / 1 / 1 | 8 |
| Chikusei | 2517 x 2335 x 128 | 2304 x 2048 x 128 | 3969 / 16 / 16 | 8 |
| CAVE | 32 scenes, each 512 x 512 x 31 | unchanged | 3600 / 4 / 12 (16 / 4 / 12 scenes) | 3 |
| Botswana | 1476 x 256 x 145 | unchanged | 269 / 1 / 1 | 8 |
| Augsburg synthetic x4 | train 540 x 1371 x 242; validation 300 x 639 x 242; test 300 x 360 x 242 | no full-cube crop; fixed tiles | 615 / 8 / 4 | 4 |

Counts use train patch=64, stride=32, evaluation patch=128, scale=4; CAVE evaluates full scenes.
Single-scene center rectangles refer to the scale-trimmed image, using zero-based half-open coordinates:
PaviaU `[240:368,106:234]`, Houston13 `[110:238,888:1016]`, Botswana `[674:802,64:192]`.
Their validation rectangle is `[0:128,0:128]`. Normalization remains the existing full-scene
min/max convention; disjoint samples do not imply training-only normalization statistics.

Chikusei MATLAB v7.3 dimensions are reversed on disk. The loader now restores `(H,W,C)`
using the MATLAB attribute, then center-crops the **raw** scene at origin `(106,143)`.
No preliminary scale trim is performed for Chikusei. Old results from the transposed HDF5
fallback or previous crop are a different protocol and must be rerun for a fair comparison.

CAVE `watercolors` is an official 8-bit RGBA exception: all 31 local PNGs were checked
byte-for-byte against the [Columbia ZIP](https://www.cs.columbia.edu/CAVE/databases/multispectral/zip/watercolors_ms.zip).
RGB components must be identical and alpha opaque; the scalar band is divided by 255.
Other local scenes are 16-bit grayscale and divided by 65535. No color averaging or alpha-as-spectrum is used.

Augsburg's 242-band metadata contains a VNIR/SWIR wavelength overlap (985 -> 905 nm).
For synthetic SRF quadrature, cell widths are computed on sorted unique centres and divided
among duplicate-centre bands, then mapped back to the original cube order. This is an explicit
centre-sampling approximation, not recovery of the complete HySpex-to-Sentinel instrument simulation.
Positive widths and row-sum checks alone are insufficient without this overlap handling.
Bundled sensor paths are resolved relative to the repository rather than the launch directory.

The three official Augsburg footprints are disjoint in EPSG:32632. Existing synthetic x4 evaluation
uses only complete 128 tiles: validation covers 256 x 512 of 300 x 639, and test covers 256 x 256
of 300 x 360 (65,536 / 108,000 pixels). Discarded border pixels are not evaluated. This is **not**
the full-region official MDAS x3 benchmark. See `AUGSBURG_REAL_WORLD_PROTOCOL.md` for the proposed
x3 full-region and real-MSI experiments; those experiments have not been run by the data audit.

Use the same degradation settings explicitly in both repositories: Both repositories and EMR-Diff now default to physical;
older comparison versions defaulted to gaussian_bicubic. Matching splits/SRF does not
make those two different observation operators equivalent. Formal comparisons must pass
`--degradation_mode physical --scale_ratio 4` (or the same explicitly selected alternative) to all methods.
New dataset choices are enabled in generic training entry points; this audit validates data pipelines,
not the memory requirements or accuracy of every model on full 512 x 512 CAVE evaluation images.

Multi-sample evaluation: the S2Diff-MH CDRDI/GIGI training evaluators now aggregate every
held-out sample (macro mean; minimum Jacobian retains the worst case). Previously they silently
used only the first validation/test patch. The legacy standalone CDRDI final-test diagnostic
and HSIFN misalignment visualization now reject multi-sample splits rather than report partial
results as full benchmarks. Model-specific visualization extensions remain separate work.
