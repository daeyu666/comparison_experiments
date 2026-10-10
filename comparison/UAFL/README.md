# UAFL — CVPR 2026 reproduction


> **Unified synthetic-x4 entry point (current formal protocol):**
>
> New/re-run registered and mixed experiments must be launched from the
> repository root:
>
> ```bash
> python train.py --model UAFL --dataset <Dataset> --mode registered
> python train.py --model UAFL --dataset <Dataset> --mode mixed
> python test.py  --model UAFL --dataset <Dataset> --mode registered
> python test.py  --model UAFL --dataset <Dataset> --mode mixed
> ```
>
> The root protocol freezes stage 2 to **10% exact identity + 90% deformed**;
> the deformed component uses `dx,dy~U(-4,4)`,
> `rotation~U(-2,2)`, and local-amplitude proposal `U(0,4)` with
> `min Jacobian>=0.5`. Model-local shell launchers below are retained for
> historical reproducibility and should not be copied into new experiments.
>
> Existing UAFL mixed checkpoints trained before this change are an explicit
> legacy exception: their stage-2 loader was deformed-only. They are not
> relabeled as 10/90 checkpoints and do not need to be retrained for the
> current time-constrained result set.


Paper: **Enhancing Unregistered Hyperspectral Image Super-Resolution via Unmixing-based Abundance Fusion Learning**, CVPR 2026.

Official implementation: `yingkai-zhang/UAFL`.

This directory contains the UAFL reproduction used by `comparison_experiments`.

## Reproduction rule

The UAFL learnable topology is kept unchanged. Only the high-resolution reference
channel count is adapted from the paper RGB input to the benchmark sensor MSI, and
obsolete low-level CUDA calls are replaced by mathematically equivalent current
PyTorch operators.

Preserved UAFL components include:

- SVD-based HSI unmixing;
- abundance-map representation;
- Coarse-to-Fine Deformable Aggregation (CFDA);
- prior-flow / similarity prediction;
- residual deformable convolution;
- Spatial-Channel Abundance Cross-Attention (SCACA);
- Spatial-Channel Modulated Fusion (SCMF);
- abundance residual prediction and endmember reconstruction.

The paper-sized reproduction uses:

```text
dim                 = 128
stage               = 1
num_blocks          = [2, 1]
num_endmembers      = 3
deformable_groups   = 8
```

The public repository does not include the exact experiment YAML; `dim=128` is an
architecture / parameter-count reconstruction consistent with the paper-reported
~5.94 M parameter regime, not a claim about an unpublished config file.

## Sensor adaptation

Only the first reference projection changes channel count:

```text
paper RGB:       Conv2d(3, dim, 3, 1, 1)
PaviaU IKONOS:   Conv2d(4, dim, 3, 1, 1)
Houston/Chikusei Conv2d(8, dim, 3, 1, 1)
```

For the formal PaviaU experiment:

```text
HSI bands = 103
MSI sensor = IKONOS
MSI bands = 4
scale = x4
```

## Formal comparison protocol — deformation on HSI acquisition

The current formal protocol is matched to `S2Diff-MH`. The reliable coordinate
system is the HR-MSI / HR-HSI ground-truth coordinate system. Misregistration is
introduced on the HSI acquisition path, not on MSI:

```text
reliable HR-HSI X
    |-- SRF R0 ------------------------------> registered HR-MSI Y_M
    |
    +-- geometry W_phi
          -> fixed physical spatial degradation P0
          -> deformed LR-HSI Y_H
```

In equations:

```text
Y_M = R0(X)
Y_H = P0(W_phi(X))
```

UAFL receives bicubic-upsampled `Y_H` and the registered HR-MSI `Y_M` and predicts
`X`. No inverse warp is applied to LR-HSI.

The physical degradation is applied **after** the HSI deformation:

```text
HR-HSI deformation
-> calibrated Gaussian PSF from MTF@LR-Nyquist=0.2
-> detector-area integration
-> x4 sampling
```

### Geometry distribution

Formal settings:

```text
translation:
  dx ~ U(-4, +4) HR pixels
  dy ~ U(-4, +4) HR pixels

rotation:
  theta ~ U(-2, +2) degrees

local non-rigid deformation:
  max_local = 4 HR pixels
  5x5 zero-mean sparse control points
  cubic B-spline interpolation to a dense field
  minimum Jacobian determinant >= 0.5
```

This translation definition is intentionally **per-axis**, matching S2Diff-MH.
It is different from the older UAFL radial-translation experiments in this
folder.

During training, local deformation strength is sampled across the full `0..4 px`
range, including a small fraction of zero-local cases. Final test cases use the
same deterministic stress-style sampler as S2Diff-MH.

### Spatial split

```text
PaviaU
HR train patch = 64x64
LR train patch = 16x16
train stride = 32
validation patch = 128x128
final test patch = center 128x128
metric region = full-frame
```

The fixed validation and center test regions remain disjoint from training.

## Training

UAFL optimizer/loss stays paper-style:

```text
optimizer = AdamW
learning rate = 1e-5
weight decay = 5e-5
batch size = 1
loss = L1
```

The formal deformation run warm-starts **model weights only** from the registered
UAFL best checkpoint and initializes a fresh AdamW optimizer for the new observation
distribution. Validation uses five fixed deformation cases on the 128x128
validation patch; model selection uses their mean full-frame PSNR.

Run:

```bash
git pull
bash comparison/UAFL/train_hsi_warp_final.sh
```

Default registered initialization:

```text
comparison/UAFL/checkpoints/physical/PaviaU/best.pth.tar
```

Formal deformation checkpoint/output:

```text
comparison/UAFL/checkpoints/hsi_warp_final/PaviaU/best.pth.tar
comparison/UAFL/logs/hsi_warp_final/PaviaU/history.csv
```

To resume an interrupted deformation run:

```bash
bash comparison/UAFL/train_hsi_warp_final.sh \
  --resume comparison/UAFL/checkpoints/hsi_warp_final/PaviaU/last.pth.tar
```

## Final test

The final test evaluates the **same deformation-trained UAFL checkpoint** under two
conditions:

```text
Registered: Y_H = P0(X)
Warp:       Y_H = P0(W_phi(X))
```

Warp is averaged over 10 deterministic cases with seed convention `seed+70000`,
matching the S2Diff-MH final test. The test patch is the center 128x128 PaviaU
patch and metrics are full-frame.

Run:

```bash
bash comparison/UAFL/test_hsi_warp_final.sh
```

Reported metrics:

```text
PSNR ↑
SSIM ↑
ERGAS ↓
SAM ↓
CC ↑
RMSE ↓
```

For direct comparison with the current result table, terminal/table `RMSE` is
reported in 8-bit units as `raw_RMSE * 255`. The JSON also stores raw `[0,1]`
RMSE.

Final JSON:

```text
comparison/UAFL/outputs/uafl_hsi_warp_final_PaviaU_seed10_cases10.json
```

## Legacy HR-MSI-warp experiments

`train.py` and the older `train_translation_*` / `finetune_*` launchers are retained
only to preserve previous experiments in which LR-HSI was fixed and HR-MSI was
warped. They are **not** the current formal protocol.

Do not mix checkpoints or scores from those legacy experiments with the current
HSI-deformed protocol.

## Compatibility notes

The original implementation used old `mmcv` DCNv2 binaries. `dcn_compat.py`
preserves the released weights, offsets, masks and deformable groups while using:

```text
torchvision.ops.deform_conv2d(..., mask=mask)
```

The deprecated thin-SVD call is replaced by:

```text
torch.linalg.svd(..., full_matrices=False)
```

Neither change alters UAFL's learnable topology.

## Formal files

```text
comparison/UAFL/
├── hsi_deformation.py          # matched HSI geometry + B-spline sampler
├── train_hsi_deformed.py       # formal deformation training
├── test_hsi_deformed.py        # Registered/Warp final full-metric test
├── train_hsi_warp_final.sh     # formal training launcher
├── test_hsi_warp_final.sh      # formal final-test launcher
├── model.py                    # UAFL topology
├── dcn_compat.py               # current-PyTorch DCNv2 backend
├── common.py
├── train.py                    # legacy HR-MSI-warp experiment entry
├── checkpoints/                # ignored
├── logs/                       # ignored
└── outputs/                    # ignored
```

## Augsburg-2 Wald real-Sentinel x3 branch (strict)

The **real-observation Wald** benchmark is separate from the above
synthetic x4 / HSI-warp protocols. It takes observed 30m EnMAP-like HSI,
real Sentinel-2 10m B2/B3/B4/B8 (aggregated to 30m for Wald), and an
observed-30m-HSI target. Full native-scale inference uses 30m HSI and
real 10m MSI with no 10m HR-HSI supervision.

Read [AUGSBURG2_WALD.md](AUGSBURG2_WALD.md) for the frozen protocol, cache
provenance requirements, training, held-out Wald test, and sub_area_2 inference
commands. Dedicated entry points:

- `augsburg2_wald_common.py`
- `train_augsburg2_wald.py`
- `infer_augsburg2_wald.py`

Do not run `train_hsi_deformed.py` for this real Wald task: it synthesizes
HSI deformations under a different x4 observation protocol.

## Formal four-dataset final test (registered vs HSI-deformed-mixed checkpoints)

Run from the `comparison_experiments` root once training has finished.
Applies to **synthetic x4** PaviaU, Houston13, Botswana and Augsburg
(`Augsburg-2 Wald` is the separate real x3 experiment).

One script processes 8 best checkpoints and writes 12 metric rows:

| Training weights | Registered test | Warp test |
|---|---|---|
| `physical/<dataset>/best.pth.tar` (registered-only) | Yes | No |
| `hsi_warp_final/<dataset>/best.pth.tar` (HSI-deformed mixed) | Yes | Yes |

```bash
git pull
python comparison/UAFL/test_four_datasets_final.py --dry_run
python comparison/UAFL/test_four_datasets_final.py \
  --data_root ./data/raw --seed 10 --cases 10 --device cuda:0
```

If you saved weights outside the default locations, set
`--registered_root /path/to/physical` and
`--mixed_root /path/to/hsi_warp_final`. In each root, the script expects
`<Dataset>/best.pth.tar`.

The tester validates each checkpoint's recorded training regime, dataset,
physical x4 degradation, and HSI-warp geometry. It evaluates **only held-out
test samples**. Warp uses 10 deterministic synthetic cases per sample with
the S2Diff-MH `seed+70000` convention. The HR-MSI stays registered in both
test groups. For multi-tile test splits (Augsburg), scores are macro-averaged
over 128x128 patches.

Machine-readable results and a human-readable 12-row comparison:
```text
comparison/UAFL/outputs/final_four_datasets/
  PaviaU_registered_seed10.json
  PaviaU_hsi_deformed_mixed_seed10.json
  ...
  UAFL_final_four_datasets_seed10.csv
```

Reported: PSNR, SSIM, ERGAS, SAM, CC, raw RMSE and RMSE*255.
Registered-trained checkpoints are **not** subjected to the Warp test,
and no training weights/checkpoints are overwritten by this evaluation.
