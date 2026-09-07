# PSRF-DiffNet comparison reproduction

This folder contains the comparison-oriented reproduction of **PSRF-DiffNet**:

> J. Qu, J. He, Y. Li, W. Dong and S. Liu, "Progressive Synergistic Registration and Fusion Diffusion Network for Unregistered Hyperspectral and Multispectral Image Fusion," IEEE Transactions on Geoscience and Remote Sensing, 2025.

Official code: https://github.com/Jiahuiqu/PSRF-DiffNet

The goal here is not to change the comparison protocol to match the released PaviaC files. Instead, the PSRF registration/fusion architecture is connected to the same observations used by every method in this repository.

## Shared comparison protocol

All data generation is delegated to the repository root:

- LR-HSI: `degradations/physical.py` (or the optional Gaussian+bicubic baseline).
- PaviaU HR-MSI: IKONOS 4-band SRF.
- Scale: x4.
- Train patches: 64x64, stride 32.
- Validation: fixed 128x128 region disjoint from test.
- Test: center 128x128 region.
- Non-registration: root `degradations/misalignment.py`, ported from S2Diff.
- Misalignment changes **only HR-MSI**; HR-HSI GT and LR-HSI stay unchanged.
- Primary translation-sweep metrics: valid-overlap PSNR/SAM using `valid_soft >= 0.999`.

Because the released PSRF fusion block binds a Conv1d to the training patch area, validation/test images larger than the 64x64 training patch are reconstructed by fixed-patch overlap-add tiling rather than changing the model shape.

## Architecture retained

The reproduction keeps the main released processing chain:

1. **CRN coarse registration**: LR-HSI / downsampled MSI cross-attention predicts an affine transform.
2. **FRN fine registration**: local 3x3 cross-modal matching predicts a pixel-wise local displacement.
3. **Complementary fusion**: MSI structure-tensor branch + HSI spectral/channel branch + learned guided filtering.
4. **Diffusion x0 prediction**: clean HR-HSI is predicted from a noisy HR-HSI state plus the registered LR-HSI and HR-MSI conditions.

## Necessary implementation corrections

The public release contains several experiment-specific hard codings or reverse-process issues. The comparison version makes the following engineering corrections while retaining the PSRF design:

- removes hard-coded `cuda:1`;
- removes fixed 102-HSI / 4-MSI / 160x160 assumptions;
- PaviaU therefore uses the repository's 103-band HSI and IKONOS 4-band MSI;
- converts FRN local {-1,0,1} **pixel offsets** to a normalized identity sampling grid before `grid_sample`;
- reverse diffusion now feeds every reconstructed state into the next step;
- accelerated evaluation uses a correct deterministic DDIM jump relation rather than applying a one-step posterior while skipping indices;
- training/validation/test observations all come from the shared comparison loader and degradation operators.

These corrections mean this is an architecture-faithful clean reproduction for controlled comparison, not a bit-identical execution of the released scripts.

## Train

Run from the repository root.

Registered sanity run:

```bash
python comparison/PSRF-DiffNet/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode registered
```

Translation-misaligned training, covering the full 0-6 px study range:

```bash
python comparison/PSRF-DiffNet/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 6 \
  --epochs 300
```

For a different training severity, change only `--translation_max_px`. `dx` and `dy` are sampled independently from `U(-d,d)` for each training sample, exactly as in the S2Diff misalignment augmentation.

The default validation / early-stopping protocol follows `comparison/README.md`:

- PaviaU validation every 20 epochs;
- Houston13 every 10 epochs;
- Chikusei every 5 epochs;
- monitor `PSNR_valid`;
- `min_delta = 0.02 dB`;
- patience = 2 validation evaluations;
- deterministic validation seed = 1234.

Best checkpoint:

```text
comparison/PSRF-DiffNet/checkpoints/<degradation_mode>/<Dataset>/best.pth.tar
```

## Translation sensitivity sweep

```bash
python comparison/PSRF-DiffNet/eval_misalignment.py \
  --checkpoint comparison/PSRF-DiffNet/checkpoints/physical/PaviaU/best.pth.tar \
  --dataset PaviaU \
  --degradation_mode physical \
  --misalignment_shifts 0 0.5 1 2 3 4 6 \
  --misalignment_trials 5 \
  --sample_steps 200
```

The script reuses the same normalized random shift direction and diffusion initialization across severity values within each trial, giving a paired curve directly comparable with the S2Diff diagnostic.

Outputs:

```text
comparison/PSRF-DiffNet/outputs/physical/PaviaU/translation_sweep.csv
comparison/PSRF-DiffNet/outputs/physical/PaviaU/translation_sweep_details.csv
```

The summary contains `PSNR_valid`, `SAM_valid`, full-frame PSNR/SAM/RMSE/ERGAS/SSIM/CC, valid fraction, actual sampled shift magnitude, and mean FRN displacement.

## Files

```text
PSRF-DiffNet/
├── network.py              # CRN + FRN + complementary fusion
├── diffusion.py            # x0 diffusion training + corrected DDIM/tiled inference
├── common.py               # shared protocol/misalignment/valid metrics helpers
├── train.py                # training + validation + early stopping
├── eval_misalignment.py    # 0/0.5/1/2/3/4/6 px paired sweep
├── smoke_test.py           # dimension/interface smoke test
└── README.md
```
