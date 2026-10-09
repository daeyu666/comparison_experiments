# EMR-Diff — legacy Augsburg-2 full-region Wald x3

> **Superseded for formal comparison.**
>
> The current S2Diff-MH / UAFL protocol is the leakage-controlled central
> holdout `Augsburg2-Wald-center-holdout-v1`. EMR-Diff has been migrated to
> the same protocol. Use:
>
> `comparison/EMR-Diff/AUGSBURG2_WALD_CENTER_HOLDOUT.md`

The earlier `augsburg2_wald` full-region experiment is retained only as
historical/debugging context. Its training region includes pixels from what is
now the central test ROI, its old radiometric calibration was fit under the
full-region protocol, and its whole-region QNR is **not** comparable with the
new held-out scores.

Do not reuse:

- `comparison/EMR-Diff/checkpoints/augsburg2_wald/`
- `comparison/EMR-Diff/logs/augsburg2_wald/`
- `comparison/EMR-Diff/outputs/augsburg2_wald/`
- `Augsburg2_Wald_radiometry.json`

for the center-heldout result table.

Current formal commands:

```bash
python comparison/EMR-Diff/train_augsburg2_wald.py --stage train --epochs 100 --monitor ref_sam --device cuda
python comparison/EMR-Diff/train_augsburg2_wald.py --stage test --device cuda
python comparison/EMR-Diff/infer_augsburg2_wald.py --center_holdout --write_tif
```

See `AUGSBURG2_WALD_CENTER_HOLDOUT.md` for the exact split, PSF guard,
radiometry, checkpoint provenance and QNR definitions.
