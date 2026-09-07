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
├── smoke_test.py          # 103/4, 16x16 -> 64x64 forward/backward test
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

## CUDA dependencies

The released source imports compiled modules directly:

```text
causal_conv1d_cuda
selective_scan_cuda
```

so the smoke test/training currently require CUDA and working causal-conv1d / Mamba selective-scan extensions compatible with the local PyTorch/CUDA environment. The copied Python folders alone do not replace those compiled extensions.

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
