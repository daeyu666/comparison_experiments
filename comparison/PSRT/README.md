# PSRT (IEEE TGRS 2023) — unified registered HSI/MSI fusion baseline

**Paper:** Shang-Qi Deng et al., *PSRT: Pyramid Shuffle-and-Reshuffle
Transformer for Multispectral and Hyperspectral Image Fusion*, IEEE TGRS, 2023.
DOI: 10.1109/TGRS.2023.3244750.

**Official repository:** https://github.com/shangqideng/PSRT
**Core source:** `model/model_SR.py` (blob SHA `07c41e4290719edc2fc1fd15ef56fec69d662d42`).
**Ported core:** `comparison/PSRT/model_SR.py`.

## Fidelity and changes from released source

The original **PSRTnet** topology is preserved:

- Input: concatenation of HR-MSI and bicubic-upsampled LR-HSI.
- Feature extractor: one 3x3 convolution to 32 feature channels.
- Exactly 2 outer blocks, each containing 3 pyramid blocks.
- Each pyramid contains W2W attention blocks at window sizes **8, 4, 2**.
- Each W2W contains attention → window shuffle → attention →
  inverse reshuffle → attention, followed by a residual connection.
- 8 heads, 32 channels, default qk scale 4; released initialization retained.
- Output: 3x3 convolution, LeakyReLU, residual addition to upsampled HSI.

Changes are limited to allowing **dynamic HSI/MSI channel counts** (the released
network unconditionally assumes 31 HSI and 3 MSI channels), configurable
64x64 patch, replacing an unused dependency on `timm.models.layers` utilities
with equivalent local defaults, and wrapping bicubic interpolation in
`model.py`. No extra cross-attention, loss, registration network or novel
module has been added.

The original repository did not include a top-level LICENSE file when checked;
the source is attributed above. This adapter is intended as a research
reproduction, not a claim that the project is a new architecture.

## Optimization and benchmark protocol

The released `main_x4.py` uses **L1 reconstruction loss**, AdamW with
learning rate `1e-4` and weight decay `1e-4`, and global gradient
clipping at 10. These are retained. Epoch budgets and data protocols use the
frozen comparison_experiments conventions instead of the released HDF5
CAVE/Harvard files.

- Common synthetic x4 input; **64x64 HR train patch / stride 32**;
  **128x128 held-out validation and test**.
- Shared **physical** LR-HSI PSF/MTF (Nyquist MTF 0.2), shared sensor SRF.
- PaviaU HSI=103 bands, MSI=4 bands. Other datasets use their configured
  HSI/MSI bands dynamically (CAVE, Chikusei, Houston13, Botswana, Augsburg).
- Stage 1 registered: `Y_H = P0(X)`, `Y_M = R0(X)`.
- Stage 2 mixed: 10% identity, 90% HSI-only geometric deformation,
  `Y_H=P0(W_phi(X))` while the MSI is held in the reference geometry.
- Geometry: HR translation ±4px, rotation ±2°, local B-spline ≤4px,
  5x5 control grid, minimum Jacobian 0.5.
- Optimizer remains **AdamW 1e-4/1e-4** (not the default 1e-5/5e-5
  optimizer of other comparison models).
- Nominal stage budgets: **800 registered + 600 mixed**, selected from the
  common frozen protocol, **not** the original paper's 2001-epoch script.
- Selection: registered PSNR (stage 1), warped PSNR (stage 2).
- Main-table mixed evaluation: 10 deterministic geometry seeds, 10 cases per
  held-out patch per seed; registered result evaluated once.
- Strict metadata checks: model stage, channels, sensor SRF fingerprint,
  architecture ID and shared data protocol.

This adaptation uses original PSRT architecture/optimization but not the
released data generation, so results should be described as a **unified
benchmark reproduction**, not direct reproduction of original paper numbers.

## Commands (from comparison_experiments root)

1. Get current code with `git pull`.
2. Verify shape, shuffle reversibility, gradients, and full-image inference:

```bash
python comparison/PSRT/smoke_test.py --device cuda:0 --full
```

3. Print the registered launch command without training:

```bash
python train.py --model PSRT --dataset PaviaU --mode registered --device cuda:0 --dry_run
```

4. Execute normal two-stage training:

```bash
python train.py --model PSRT --dataset PaviaU --mode registered --device cuda:0
python train.py --model PSRT --dataset PaviaU --mode mixed --device cuda:0
```

Resume an existing stage without restarting optimizer/RNG:

```bash
python train.py --model PSRT --dataset PaviaU --mode registered --device cuda:0 --resume auto
```

5. Registered test and mixed main-table 10x10 test:

```bash
python test.py --model PSRT --dataset PaviaU --mode registered --device cuda:0
python test.py --model PSRT --dataset PaviaU --mode mixed --device cuda:0
```

Mixed quick check (do not use smoke results in publication tables):

```bash
python test.py --model PSRT --dataset PaviaU --mode mixed \
  --device cuda:0 --geometry_seeds 10001 --cases 1 --max_samples 1
```

Training smoke with **separate checkpoint/log folders**:

```bash
python comparison/PSRT/train.py --dataset PaviaU --epochs 1 \
  --max_train_batches 1 --device cuda:0 \
  --checkpoint_dir comparison/PSRT/checkpoints/SMOKE/PaviaU \
  --log_dir comparison/PSRT/logs/SMOKE/PaviaU
```

## Outputs

```text
comparison/PSRT/checkpoints/physical/<Dataset>/best.pth.tar
comparison/PSRT/checkpoints/physical/<Dataset>/last.pth.tar
comparison/PSRT/checkpoints/hsi_warp_final/<Dataset>/best.pth.tar
comparison/PSRT/checkpoints/hsi_warp_final/<Dataset>/last.pth.tar
comparison/PSRT/logs/<physical|hsi_warp_final>/<Dataset>/validation_history.csv
comparison/PSRT/outputs/multiseed_deformation/<Dataset>/
```

**Status:** The GitHub adapter and static integration checks are done.
GPU smoke execution, benchmark training and actual PSNR/SAM verification must
be performed in the server environment. The registered and mixed checkpoints
do not exist until the corresponding training finishes.
