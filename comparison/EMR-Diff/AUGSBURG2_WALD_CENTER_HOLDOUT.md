# EMR-Diff Augsburg-2 center-heldout Wald x3

This is the current **real-world comparison protocol** for EMR-Diff.
It matches the S2Diff-MH / UAFL center-heldout Augsburg-2 protocol and
must be trained from scratch. Old full-region Wald checkpoints are invalid.

## Shared observed data and spatial split

Use the cache produced by S2Diff-MH
`prepare_augsburg2_wald_center_holdout.py`.

- Original Region-2 observations:
  - HSI30: 100x120x242 observed EnMAP-like HSI.
  - MSI10: 300x360x4 real Sentinel-2 B2/B3/B4/B8.
- Held-out test rectangle on the 30m HSI grid:
  - `[row24:72, col36:84]`, shape 48x48x242.
- Same native test rectangle on the 10m MSI grid:
  - `[row72:216, col108:252]`, shape 144x144x4.
- Forbidden training rectangle:
  - `[row18:78, col30:90]` on the 30m grid.
  - This is the heldout region plus a 6-pixel (30m) Gaussian-PSF guard.
- Train target: observed HSI30 **outside** the forbidden rectangle.
- LR-HSI: fixed Wald Gaussian sigma=1.2 / stride3, i.e. HSI90.
- Train MSI: real S2 MSI10 area-averaged to MSI30.
- Training patches: 24x24, stride 6.
- Every patch intersecting the forbidden rectangle is hard rejected.
- No random flip/rotation.
- Validation: geographically disjoint original `deep_valid`, 48x48 blocks.
- RR test: only the heldout center 48x48 HSI30 target with 16x16 HSI90 and
  48x48 real-MSI30 inputs.
- Native inference: only the untrained 144x144 center MSI10 rectangle.
- There is no HSI10 ground truth; native inference reports QNR/Dlambda/Ds only.

The split ID is frozen as:

```text
Augsburg2-Wald-center-holdout-v1
```

EMR-Diff checkpoints store this ID and the 30m test bbox. A legacy
full-region checkpoint is rejected.

## 1. Prepare the shared cache and center-only calibration

Run once in the sibling S2Diff-MH repository:

```bash
cd ../S2Diff-MH
git pull origin main

python prepare_augsburg2_wald_center_holdout.py \
  --source_wald_root ./data/augsburg2_wald \
  --output_root ./data/augsburg2_wald_center_holdout \
  --test_size_30m 48 --guard_30m 6 \
  --train_patch_30m 24 --train_stride_30m 6

python calibrate_augsburg2_wald_radiometry.py \
  --wald_root ./data/augsburg2_wald_center_holdout \
  --real_cache_root ./data/augsburg_real_cache \
  --output ./data/calibration/Augsburg2_Wald_center_holdout_radiometry.json
```

The new radiometry **must** be fitted without the center ROI and its PSF guard.
Do not reuse `Augsburg2_Wald_radiometry.json`.

Then return to comparison_experiments:

```bash
cd ../comparison_experiments
git pull origin main
```

## 2. Train EMR-Diff from scratch

The dedicated EMR runner now defaults to the center-heldout cache,
new calibration, 24x24 training patches, and isolated output directories:

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py --stage train --epochs 100 --monitor ref_sam --device cuda
```

Equivalent fully explicit command:

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py \
  --stage train \
  --wald_root ../S2Diff-MH/data/augsburg2_wald_center_holdout \
  --radiometry_json ../S2Diff-MH/data/calibration/Augsburg2_Wald_center_holdout_radiometry.json \
  --checkpoint_dir comparison/EMR-Diff/checkpoints/augsburg2_wald_center_holdout \
  --log_dir comparison/EMR-Diff/logs/augsburg2_wald_center_holdout \
  --train_patch_size 24 --train_stride 6 --eval_patch_size 48 \
  --min_valid_fraction 0.8 --batch_size 1 \
  --lr 1e-5 --weight_decay 5e-5 \
  --epochs 100 --eval_interval 5 --save_interval 5 \
  --monitor ref_sam --seed 10 --eval_seed 1234 --device cuda
```

Outputs:

- `comparison/EMR-Diff/checkpoints/augsburg2_wald_center_holdout/best.pth.tar`
- `comparison/EMR-Diff/checkpoints/augsburg2_wald_center_holdout/last.pth.tar`
- `comparison/EMR-Diff/logs/augsburg2_wald_center_holdout/history.csv`

Model selection is the minimum pooled masked validation SAM, matching UAFL.
The heldout center test arrays are never opened during training.

The EMR-specific architecture adaptation remains:
242 HSI + 4 MSI = 246 diffusion-state channels, with a 64-channel default
BAFUNet latent trunk and a 1x1 state projection. For 24x24 train patches,
only internal network tensors are padded to 32x32; the loss/mask remains
restricted to the original observed 24x24 pixels.

## 3. Reduced-resolution heldout test

Observed HSI30 is available here, so this step reports reference metrics:

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py --stage test --device cuda
```

The command loads the center-trained `best.pth.tar` and reports:

```text
REF_PSNR
REF_SAM
REF_RMSE
```

The test rectangle is exactly 30m `[24:72,36:84]`.

## 4. Native heldout inference and QNR

Run only after the center-heldout best checkpoint exists:

```bash
python comparison/EMR-Diff/infer_augsburg2_wald.py --center_holdout --write_tif
```

Only the 144x144 native center ROI is reconstructed.

Outputs:

- `comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_EMRDiff_heldout_HSI.npy`
- `comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout/Augsburg2_Wald_EMRDiff_heldout_HSI.tif`
- `comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout/EMRDiff_Wald_heldout_QNR.json`
- `comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout/EMRDiff_Wald_heldout_protocol.json`

The GeoTIFF transform includes the center ROI's native 10m offset.

QNR uses the **same UAFL implementation**:

- spectral reference: observed 242-band HSI30 center ROI;
- spatial reference: original real Sentinel-2 MSI10 center ROI;
- Dlambda: all 242 HSI bands / 29161 band pairs;
- Ds: HSI bands within each S2 channel's >=1% peak SRF support;
- 48x48 high-resolution UIQI windows / 16x16 low-resolution windows;
- minimum valid fraction 0.8;
- no PAN, no synthetic HSI10 reference, no HSI->4-band projection for QNR.

## 5. Regression checks

From comparison_experiments root:

```bash
python -m unittest discover -s comparison/EMR-Diff -p "test_augsburg2_wald_emr.py"
```

Optional slower CPU backbone shape test:

```bash
EMR_WALD_TEST_MODEL=1 python -m unittest discover -s comparison/EMR-Diff -p "test_augsburg2_wald_emr.py"
```

## Do not mix these results

Do **not** compare or reuse:

- `comparison/EMR-Diff/checkpoints/augsburg2_wald/`
- `comparison/EMR-Diff/outputs/augsburg2_wald/`
- old full-region `Augsburg2_Wald_radiometry.json`
- old whole-region QNR
- any synthetic Augsburg x4 checkpoint

as center-heldout results. The current official real-world comparison is the
center-heldout v1 protocol only.
