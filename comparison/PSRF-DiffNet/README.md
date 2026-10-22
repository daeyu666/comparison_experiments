# PSRF-DiffNet comparison adapter

This directory adapts **PSRF-DiffNet** to the shared synthetic comparison protocol.

> J. Qu, J. He, Y. Li, W. Dong and S. Liu, “Progressive Synergistic Registration and Fusion Diffusion Network for Unregistered Hyperspectral and Multispectral Image Fusion,” *IEEE Transactions on Geoscience and Remote Sensing*, 2025.

- Official repository: https://github.com/Jiahuiqu/PSRF-DiffNet
- The code is an architecture-aligned port of the active released execution path, not a bit-identical reproduction of its fixed-device/fixed-shape scripts.

## Active model path and fidelity notes

The active model path in the released implementation is:

1. **Coarse Registration Network (CRN)** — shared HSI/MSI token normalization, cross-modal multi-head attention and affine-transform regression.
2. **Fine Registration Network (FRN)** — local 3x3 cross-modal candidate matching and pixel-wise displacement sampling.
3. **Complementary Fusion** — MSI structure-tensor spatial guidance, HSI spectral/channel attention, learned guided filtering and residual reconstruction.
4. **Diffusion x0 prediction** — reconstructs clean HR-HSI from the current noisy HR-HSI state and the coarse-registered HSI condition, guided by HR-MSI.

The official source defines an SGPD module, but the released `Diffusion` wrapper instantiates and calls `CCFnet`; SGPD is not part of that active execution path. This adapter therefore does not insert SGPD into the model.

The adapter preserves the active CRN → FRN → complementary-fusion topology while making HSI/MSI channel counts, scale ratio, patch size and device configurable. It also aligns the active CRN/FRN QKV projections and shared CRN normalization more closely with the released code.

Engineering corrections that intentionally differ from the released script:

- Removes hard-coded `cuda:1`, 102-HSI-band, 4-MSI-band and 160x160 assumptions.
- Keeps the official training behavior of applying CRN to upsampled LR-HSI; during reverse diffusion, applies the inferred affine transform to the **current diffusion state** at each step.
- Converts FRN's discrete local pixel displacement to a normalized sampling grid and uses explicit `(dx,dy)` ordering rather than passing raw offsets as absolute `grid_sample` coordinates.
- Uses a deterministic DDIM jump relation for accelerated evaluation and correctly feeds each reconstructed state into the next reverse step.
- Uses the repository's shared sensor SRFs, physical degradation operators, data splits and metric functions.

These changes mean the model is architecture-aligned for controlled comparison, not bit-for-bit identical to the original implementation. The CRN/FRN QKV projections now use the source implementation's bias-free linear layers; **all earlier PSRF checkpoints must be retrained**. New checkpoints record architecture ID `PSRF-DiffNet-official-active-path-v4`.

## Unified synthetic comparison protocol

The root `experiment_protocol.py` defines the formal registered/mixed stages shared with EMR-Diff and CLSNet.

- Scale: x4.
- Train patch: 64x64; stride: 32.
- Validation/test image size: 128x128, using the repository's fixed splits.
- LR-HSI: shared physical degradation (`mtf_nyquist=0.2`, `psf_truncate=3.0`).
- HR-MSI: the dataset's configured sensor SRF; this does not change between stages.
- Stage 1, registered: `Y_H = P0(X)`, `Y_M = R0(X)`, target `X`.
- Stage 2, mixed: 10% exact identity samples and 90% deformed HSI observations `Y_H = P0(W_phi(X))`; HR-MSI remains in the reliable coordinate system.
- Geometry: translation bounds ±4 HR pixels per axis, rotation bounds ±2 degrees, 5x5 B-spline control grid, local displacement up to 4 HR pixels, minimum local Jacobian 0.5.
- Optimizer: AdamW, learning rate 1e-5, weight decay 5e-5, batch size 1.
- Epoch budget: 800 registered + 600 mixed.
- Validation interval: dataset-aware values from `DATASET_VALIDATION_INTERVAL`; monitor registered PSNR in stage 1 and deformed-observation PSNR in stage 2.
- Diffusion: 2000 training timesteps; validation/main-table inference uses 50 DDIM steps by default.

## Train and test from the repository root

Train the registered checkpoint:

```bash
python train.py --model PSRF-DiffNet --dataset PaviaU --mode registered --device cuda:0
```

Train the mixed checkpoint, initializing from the registered best checkpoint:

```bash
python train.py --model PSRF-DiffNet --dataset PaviaU --mode mixed --device cuda:0
```

Run the formal main-table test:

```bash
python test.py --model PSRF-DiffNet --dataset PaviaU --mode mixed --device cuda:0
```

The main-table mixed test uses the shared 10 geometry seeds × 10 cases protocol. A small pipeline check can be run with:

```bash
python test.py --model PSRF-DiffNet --dataset PaviaU --mode mixed --device cuda:0 --max_samples 1 --geometry_seeds 10001 --cases 1
```

Registered and mixed artifacts are isolated under:

```text
comparison/PSRF-DiffNet/checkpoints/physical/<Dataset>/best.pth.tar
comparison/PSRF-DiffNet/checkpoints/hsi_warp_final/<Dataset>/best.pth.tar
comparison/PSRF-DiffNet/logs/<stage>/<Dataset>/
comparison/PSRF-DiffNet/outputs/<stage>/<Dataset>/
```

The current adapter covers the six shared synthetic datasets. A real-world Augsburg Wald training/inference adapter is not yet implemented for PSRF-DiffNet.

## Separate MSI-warp sensitivity diagnostic

`eval_misalignment.py` remains available as a **separate diagnostic** that warps HR-MSI and reports valid-overlap PSNR/SAM. It is not the formal mixed-stage protocol used by the root EMR-Diff/CLSNet comparison launcher.

```bash
python comparison/PSRF-DiffNet/eval_misalignment.py \
  --checkpoint comparison/PSRF-DiffNet/checkpoints/physical/PaviaU/best.pth.tar \
  --dataset PaviaU \
  --degradation_mode physical \
  --misalignment_shifts 0 0.5 1 2 3 4 6 \
  --misalignment_trials 5 \
  --sample_steps 50
```

This diagnostic reports valid-overlap PSNR/SAM with `valid_soft >= 0.999`, plus full-frame PSNR/SAM/RMSE/ERGAS/SSIM/CC and the estimated CRN/FRN displacement diagnostics.
