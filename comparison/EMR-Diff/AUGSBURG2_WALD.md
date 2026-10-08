# EMR-Diff — Augsburg-2 strict Wald x3, real Sentinel-2

This is an **independent** real-world experiment track under
`comparison/EMR-Diff/`. It does not change the synthetic PaviaU/Houston13/
Chikusei/CAVE/Botswana/Augsburg `Train.py` and `Test.py` workflows.

## Protocol: identical observed data and test splits to UAFL

The reference protocol is `comparison/UAFL/AUGSBURG2_WALD.md`.
Both methods use exactly the same pre-generated S2Diff-MH Wald cache,
train-only radiometry, split definitions, evaluation masks, and reference metric
implementation. No new sensor observations or pseudo 10m HSI labels are made.

- HSI: **242 observed EnMAP-like bands at 30m** are the only supervised GT.
- LR-HSI (90m) is obtained from those observations using the cached Wald x3
  operator (effective Gaussian sigma 1.2 on the 30m grid).
- MSI during Wald train/test is **real** Sentinel-2 B2/B3/B4/B8 originally
  measured at 10m and area-aggregated to the 30m Wald grid.
- The original-scale 10m inference uses observed 30m HSI + **observed, not
  shifted** Sentinel-2 10m MSI. Apply only the same train-derived 4-channel
  gain/bias from `Augsburg2_Wald_radiometry.json`.
- No geometric registration/warping, no EnMAP10 reference, no synthetic x4.
- Train tiles **72x72**, stride **6**; validation/test tiles **48x48**;
  valid fraction >= **0.8**; no spatial train augmentation.
- Train 100 epochs, AdamW, LR 1e-5, weight decay 5e-5, batch size 1,
  validation every 5 epochs, save every 5 epochs, seed 10.
- Best checkpoint is selected by **minimum pooled masked validation SAM**
  (`--monitor ref_sam`), matching UAFL. `ref_psnr` is an explicit
  alternative but must not be mixed with SAM-monitored results.
- Validation and held-out Wald test use precisely the pooled masked
  `metrics_sums` / `metrics_from_sums` from UAFL. All valid pixels are
  pooled before PSNR and SAM are computed, not averaged per tile.
- Diffusion validation uses eval seed **1234** under isolated RNG state.
- **Train never reads the held-out test arrays**.

The comparison is under the same **data protocol**, not the same method-specific
loss: EMR-Diff keeps its five-step residual diffusion, edge modulation, BAFUNet,
pseudo-MSI target (first four HSI bands), and multiscale residual supervision.
For 242+4=246 state channels, a 64-channel default BAFUNet trunk with a 1x1
projection back to 246 channels avoids scaling the full heavy 7x7-convolution
backbone to width 246. This is a documented adaptation to 242-band input,
not the original full-width baseline. `--model_width 32` is available if GPU
memory is insufficient; **do not compare different model widths as one run**.
For 72x72 crops the EMR network tensors are padded to 80x80 internally; padded
pixels are excluded from masked training loss and validation metrics. Inputs,
observed spatial grid, crop, and GT are never resized to change the Wald task.

## Preparation (run from comparison_experiments root)

Create/cache data and radiometry using the unchanged S2Diff-MH
`prepare_augsburg2_wald.py` workflow used by UAFL. Use the same cache;
never overwrite it with an inferred/synthetic replacement.

With the repositories as siblings:

```bash
git pull origin main
export WALD_ROOT="../S2Diff-MH/data/augsburg2_wald"
export WALD_RAD="../S2Diff-MH/data/calibration/Augsburg2_Wald_radiometry.json"
test -f "$WALD_ROOT/train/meta.json"
test -f "$WALD_ROOT/wald_psf.json"
test -f "$WALD_ROOT/full/meta.json"
test -f "$WALD_RAD"
```

## Train (same defaults and monitor as UAFL)

One-line command to avoid shell backslash mistakes:

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py --wald_root "$WALD_ROOT" --radiometry_json "$WALD_RAD" --stage train --train_patch_size 72 --train_stride 6 --eval_patch_size 48 --min_valid_fraction 0.8 --batch_size 1 --lr 1e-5 --weight_decay 5e-5 --epochs 100 --eval_interval 5 --save_interval 5 --monitor ref_sam --seed 10 --device cuda
```

Checkpoints:

- `comparison/EMR-Diff/checkpoints/augsburg2_wald/best.pth.tar`
- `comparison/EMR-Diff/checkpoints/augsburg2_wald/last.pth.tar`

Log: `comparison/EMR-Diff/logs/augsburg2_wald/history.csv`.

For OOM, train from scratch with `--model_width 32`, record the change, and
also retain the default 64-width setting in the reported experimental protocol.
Never resume a checkpoint of a different width.

Resume (epochs is the absolute finishing epoch, as in UAFL):

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py --wald_root "$WALD_ROOT" --radiometry_json "$WALD_RAD" --epochs 150 --monitor ref_sam --resume comparison/EMR-Diff/checkpoints/augsburg2_wald/last.pth.tar
```

## Held-out Wald x3 test (reference metrics)

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py --stage test --wald_root "$WALD_ROOT" --radiometry_json "$WALD_RAD" --checkpoint comparison/EMR-Diff/checkpoints/augsburg2_wald/best.pth.tar
```

This is the **only** fully reference-based test. Prints `EMR_WALD_TEST`
with pooled masked REF_PSNR / REF_SAM (degrees) / REF_RMSE.

## Original-resolution x3 full inference (30m HSI + 10m MSI)

```bash
python comparison/EMR-Diff/infer_augsburg2_wald.py --wald_root "$WALD_ROOT" --radiometry_json "$WALD_RAD" --checkpoint comparison/EMR-Diff/checkpoints/augsburg2_wald/best.pth.tar --tile_size 96 --tile_stride 48 --write_tif
```

Outputs:

- `comparison/EMR-Diff/outputs/augsburg2_wald/Augsburg2_Wald_EMRDiff_full_HSI.npy`
- optionally same base name with `.tif` and original spatial georeferencing
- `comparison/EMR-Diff/outputs/augsburg2_wald/EMRDiff_Wald_full_QNR.json`
- `comparison/EMR-Diff/outputs/augsburg2_wald/EMRDiff_Wald_full_protocol.json`

**Full-resolution 10m ground-truth HSI does not exist**. Never compute or
report 10m PSNR, SAM, or registration error against imaginary reference labels.

### Full 10m no-reference HSI–MSI QNR

The full inference invokes UAFL's **same**
`comparison/UAFL/augsburg2_wald_qnr.py:evaluate_cache`, with:

- spectral reference: **observed 242-band 30m LR-HSI**;
- spatial reference: **real original 10m HR-MSI** with train-only gain/bias;
- Dlambda: all 242 HSI bands (29161 spectral pairs);
- Ds: only HSI bands covered by each Sentinel-2 SRF channel at 1% of peak;
- masked non-overlapping 48x48 (10m) and 16x16 (30m) UIQI windows,
  minimum valid fraction 0.8;
- QNR = max(0,1-Dlambda) * max(0,1-Ds).

**No PAN, no 4-band HSI projection for quality measurement, no geometry warp**.
Default `--qnr_window_hr 48 --qnr_min_valid_fraction 0.8
--qnr_support_fraction 0.01` exactly matches UAFL.

Re-evaluate an already-generated `.npy` without retraining:

```bash
python comparison/UAFL/augsburg2_wald_qnr.py --wald_root "$WALD_ROOT" --radiometry_json "$WALD_RAD" --fused comparison/EMR-Diff/outputs/augsburg2_wald/Augsburg2_Wald_EMRDiff_full_HSI.npy --output_json comparison/EMR-Diff/outputs/augsburg2_wald/EMRDiff_Wald_full_QNR.json
```

## Provenance and safeguards

The runner checks strict cache metadata, Wald sigma, 4x242 SRF weights,
calibration SHA256, expected field dimensions and checkpoint method/protocol/
monitor/width. The calibration is **not fitted** from the held-out test or
unavailable 10m HSI reference. Do not load synthetic Augsburg x4 or any
other UAFL/S2Diff-MH checkpoint here.
