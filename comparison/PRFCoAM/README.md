# PRFCoAM reproduction

Paper: **A Progressive Registration-Fusion Co-Optimization A-Mamba Network: Toward Deep Unregistered Hyperspectral and Multispectral Fusion**, IEEE TGRS, 2025.

This directory keeps the author's released implementation under `base/` and adds a thin benchmark adaptation layer at the PRFCoAM root. The current phase is deliberately **registered-only**: first verify that the published PRFCoAM topology learns correctly on the shared PaviaU physical-degradation benchmark, then adapt the registration direction for the repository-wide HR-MSI-only misalignment protocol.

## Layout

```text
comparison/PRFCoAM/
├── base/                 # author's released source, preserved
├── common.py             # shared comparison_experiments config helpers
├── model_adapter.py      # dynamic channels/device patch only
├── train.py              # Phase-A registered reproduction
├── smoke_test.py         # 103/4, 16x16 -> 64x64 forward/backward test
├── env_check.py          # CUDA extension / ABI diagnosis
├── checkpoints/
├── logs/
└── outputs/
```

## What is preserved from the released code

- `Net` topology from `base/model_ssm_fuse9_2.py`.
- two local-aware registration stages.
- two progressive x2 fusion stages (overall x4).
- custom spatial/spectral Mamba paths.
- released objective:

```text
1.1 * L1(HR-HSI, GT-HSI)
+ 0.1 * L1(R(HR-HSI), HR-MSI)
+ 0.01 * displacement_smoothness
```

- Adam with initial learning rate `5e-4`.
- `StepLR(step_size=100, gamma=0.8)`.

## Benchmark adaptations

The author's PaviaC code hard-codes `102` HSI bands, `4` MSI bands, `40x40` LR-HSI and `160x160` HR outputs. The adapter changes only dataset-specific assumptions:

- PaviaU: `103` HSI bands + IKONOS `4`-band SRF MSI.
- Houston13 / Chikusei: dynamic HSI channels + WorldView-2 `8`-band MSI.
- shared x4 data loader: `64x64` HR train patch -> `16x16` LR-HSI.
- shared `physical` or `gaussian_bicubic` LR-HSI degradation.
- fixed disjoint `128x128` validation region.
- dataset-aware validation interval and validation-PSNR early stopping.
- actual runtime CUDA device replaces the author's hard-coded `cuda:1` spatial-transform device.

`base/` itself is not edited.

## CUDA dependencies and ABI compatibility

The released source bundles **Mamba 1.0.1-era Python code** and imports these compiled modules directly:

```text
causal_conv1d_cuda
selective_scan_cuda
```

The bundled causal wrapper uses the legacy four-argument CUDA call:

```text
causal_conv1d_fwd(x, weight, bias, activation)
```

Therefore installing the newest causal-conv1d blindly is not recommended: recent releases changed the CUDA-call signature. For the author's code, the compatibility target is:

```text
mamba-ssm == 1.0.1
causal-conv1d == 1.0.2
```

Both CUDA extensions must be built against the **same PyTorch installation that will run PRFCoAM**. A stale wheel built against another torch version commonly fails with an undefined C10 symbol such as `torchCheckFail`.

### Diagnose first

```bash
python comparison/PRFCoAM/env_check.py
```

The important values are:

```text
torch version
torch CUDA runtime
nvcc version
torch CXX11 ABI
causal-conv1d package version
mamba-ssm package version
causal_conv1d_cuda import
selective_scan_cuda import
```

`torch.version.cuda` and the local `nvcc --version` should be compatible. If the current extensions fail to import, rebuild the legacy-compatible pair from source against the current torch.

### Rebuild the legacy-compatible CUDA extensions

From the same conda environment used for training:

```bash
pip uninstall -y causal-conv1d mamba-ssm
pip install -U packaging ninja wheel setuptools

CAUSAL_CONV1D_FORCE_BUILD=TRUE \
pip install --no-build-isolation --no-cache-dir --no-binary=:all: \
  causal-conv1d==1.0.2

MAMBA_FORCE_BUILD=TRUE \
pip install --no-build-isolation --no-cache-dir --no-binary=:all: \
  mamba-ssm==1.0.1
```

Then verify the compiled modules directly:

```bash
python comparison/PRFCoAM/env_check.py
```

Both should report `IMPORT OK`:

```text
causal_conv1d_cuda: IMPORT OK
selective_scan_cuda: IMPORT OK
```

If source compilation fails before CUDA compilation starts, check that `nvcc` exists and that its CUDA major version is compatible with `torch.version.cuda`.

## Step 1: smoke test

From the repository root:

```bash
python comparison/PRFCoAM/smoke_test.py
```

Expected end of output:

```text
PRFCoAM smoke test: PASS
pred=(1, 103, 64, 64)
pred_msi=(1, 4, 64, 64)
```

This test performs both forward and backward and checks finite gradients.

## Step 2: PaviaU registered sanity run

```bash
python comparison/PRFCoAM/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --epochs 300
```

Defaults relevant to the shared benchmark:

```text
HR train patch  = 64x64
LR train patch  = 16x16
stride          = 32
scale           = x4
batch size      = 4
Adam lr         = 5e-4
validation      = every 20 epochs on PaviaU
early-stop      = min_delta 0.02 dB, patience 2 validations
```

If memory is insufficient, reduce only the optimization batch size, e.g. `--batch_size 1`; spatial patch size should remain `64` for the shared comparison.

Best checkpoint:

```text
comparison/PRFCoAM/checkpoints/physical/PaviaU/best.pth.tar
```

Training history:

```text
comparison/PRFCoAM/logs/physical/PaviaU/history.csv
```

## Important protocol note

The published PRFCoAM data protocol deforms **LR-HSI** (`LRHS_alpha*`) while keeping HR-MSI and GT-HSI fixed. The repository-wide S2Diff robustness protocol instead keeps LR-HSI/GT fixed and perturbs **HR-MSI**. Therefore the current `train.py` intentionally uses registered inputs only.

After the registered reproduction is validated, Phase B will adapt the MULAR registration direction to align perturbed HR-MSI toward the fixed HSI/GT frame, while keeping the shared misalignment generator and valid-overlap metrics unchanged.
