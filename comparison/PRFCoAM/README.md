# PRFCoAM reproduction

Paper: **A Progressive Registration-Fusion Co-Optimization A-Mamba Network: Toward Deep Unregistered Hyperspectral and Multispectral Fusion**, IEEE TGRS, 2025.

This directory keeps the author's released implementation under `base/` and adds a thin benchmark adaptation layer at the PRFCoAM root. The current phase is deliberately **registered-only**: first verify that the published PRFCoAM topology learns correctly on the shared PaviaU physical-degradation benchmark, then adapt the registration direction for the repository-wide HR-MSI-only misalignment protocol.

## Layout

```text
comparison/PRFCoAM/
├── base/                 # author's released source, preserved
├── common.py             # shared benchmark helpers
├── mamba_compat.py       # self-contained author four-scan Mamba backend
├── model_adapter.py      # dynamic channels/device/backend injection
├── train.py              # Phase-A registered reproduction
├── smoke_test.py         # 103/4, 16x16 -> 64x64 forward/backward test
├── env_check.py          # Torch / self-contained scan diagnosis
├── checkpoints/
├── logs/
└── outputs/
```

## What is preserved from the released code

- `Net` topology from `base/model_ssm_fuse9_2.py`.
- two local-aware registration stages.
- two progressive x2 fusion stages (overall x4).
- released v2 spectral bidirectional scan and v3 spatial four-direction scan.
- channel/spatial attention branches.
- SSM A/B/C/D parameterization, delta projection, SiLU gate and output projection.
- released objective:

```text
1.1 * L1(HR-HSI, GT-HSI)
+ 0.1 * L1(R(HR-HSI), HR-MSI)
+ 0.01 * displacement_smoothness
```

- Adam with initial learning rate `5e-4`.
- `StepLR(step_size=100, gamma=0.8)`.

## Benchmark adaptations

The author's PaviaC code hard-codes `102` HSI bands, `4` MSI bands, `40x40` LR-HSI and `160x160` HR outputs. The adapter changes dataset/environment-specific assumptions:

- PaviaU: `103` HSI bands + IKONOS `4`-band SRF MSI.
- Houston13 / Chikusei: dynamic HSI channels + WorldView-2 `8`-band MSI.
- shared x4 data loader: `64x64` HR train patch -> `16x16` LR-HSI.
- shared `physical` or `gaussian_bicubic` LR-HSI degradation.
- fixed disjoint `128x128` validation region.
- dataset-aware validation interval and validation-PSNR early stopping.
- actual runtime CUDA device replaces the author's hard-coded `cuda:1` spatial-transform device.
- the released Mamba-1.0.1 fused CUDA ABI is replaced by a self-contained PyTorch execution backend.

`base/` itself is not edited.

## Self-contained Mamba compatibility backend

The released source directly depends on old `causal_conv1d_cuda` and `selective_scan_cuda` interfaces. On the target Torch 2.6.0 + cu124 environment, both old-source builds and modern prebuilt Mamba wheels can fail with C++/libtorch undefined-symbol errors.

PRFCoAM therefore no longer imports any installed `mamba_ssm` package. `model_adapter.py` injects `mamba_compat.py` under the module name expected by the author's source, while preserving the released v2/v3 scan logic.

The compatibility backend replaces only the execution kernels:

```text
released depthwise causal conv CUDA
    -> native PyTorch grouped causal conv1d

released selective_scan_cuda
    -> native PyTorch parallel affine-prefix selective scan
```

The selective state update is still the same recurrence used by Mamba:

```text
state_t = exp(delta_t * A) * state_(t-1) + delta_t * B_t * u_t
```

Instead of iterating over every sequence position in Python, the recurrence is evaluated as an associative affine-prefix scan in approximately `log2(L)` tensor rounds. This keeps the implementation independent of binary extensions without reducing the model to a slow Python 4096-step loop.

No `mamba-ssm` or `causal-conv1d` installation is required for this reproduction. Existing broken installations may be removed without touching PyTorch:

```bash
python -m pip uninstall -y mamba-ssm causal-conv1d
```

Then run:

```bash
python comparison/PRFCoAM/env_check.py
```

The key end of the output should be:

```text
self-contained selective scan numerical check: OK
self-contained selective scan forward/backward: OK
external Mamba CUDA extensions required: NO
PRFCoAM backend status: READY
```

## Step 1: smoke test

From repository root:

```bash
python comparison/PRFCoAM/smoke_test.py
```

Expected end of output:

```text
PRFCoAM smoke test: PASS
pred=(1, 103, 64, 64)
pred_msi=(1, 4, 64, 64)
```

This performs forward, the released objective, backward and finite-gradient checks.

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

The published PRFCoAM data protocol deforms **LR-HSI** while keeping HR-MSI and GT-HSI fixed. The repository-wide S2Diff robustness protocol instead keeps LR-HSI/GT fixed and perturbs **HR-MSI**. Therefore the current `train.py` intentionally uses registered inputs only.

After the registered reproduction is validated, Phase B will adapt the registration direction to align perturbed HR-MSI toward the fixed HSI/GT frame, while keeping the shared misalignment generator and valid-overlap metrics unchanged.
