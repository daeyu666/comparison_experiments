# SA-RFM — unified HSI/MSI fusion comparison adapter

**Method:** Structure-Aware Riemannian Flow Matching for Hyperspectral and
Multispectral Image Registration and Fusion (ICML 2026).

- Official implementation: https://github.com/ZhangQuan-hub/SA-RFM
- Released core: `SB/sb.py` (blob SHA `bb1f9c472d9539c73c4f3bd3d296990b8e16b2de`)
- Mirrored core: `upstream_sb.py` (only an attribution header added)
- Upstream license: `LICENSE.upstream` (MIT; copyright ZhangQuan-hub 2026)
- Comparison wrapper: `model.py` (`SARFMModel`).

## Architecture and fidelity

The model retains the official `ImprovedDiT` implementation from `SB/sb.py`.
Despite the name, this released network is a convolutional velocity predictor:
shared spectral/HSI encoders, MSI encoder, time embedding, structure tensor /
Riemannian metric, grouped convolution fusion, and spectral velocity output.
There is **no repeated spatial downsampling**, so 64x64 training patches are
supported without a 1x1 spatial bottleneck.

The official training script instantiates:

- `ImprovedDiT(hsi_channels=N, msi_channels=5, dim=220, alpha=1, beta=10, sigma=1)`.
- `AdamW(lr=1e-4, weight_decay=1e-4, betas=(0.9, 0.999))`.
- Effective loss `L_RFM + lambda_otd * L_OTD`, with CLI default
  `lambda_otd=1.0`.
- **Euler ODE inference with one step** (the official `--steps` default is 1).

The official `SA_SB_Trainer` calculates gradient consistency and spectral angle
diagnostics but its actual `total_loss` expression **does not add those terms**.
We reproduce the *actual optimized objective*, rather than silently adding
terms that are not active in the released training path.

An Euler wrapper reimplements the same update `x += v / steps` and does not
copy every intermediate state to CPU. It defaults to 1 step in all formal tests;
other step counts are explicit diagnostic overrides. The core model topology,
Riemannian loss, and OTD loss come from the upstream snapshot unchanged.

## Comparison protocol

Registered and mixed stages use the common `comparison_experiments` synthetic
protocol, NOT the upstream dataset preparation:

- HR training patch **64x64**, stride 32; validation/test **128x128**.
- Scale x4; shared physical MTF degradation (Nyquist MTF 0.2; truncate 3.0).
- Sensor-specific MSI SRF and channel counts (e.g. **PaviaU 103/4**,
  Chikusei 128/8, Houston13 144/8), not the source's hard-coded 5 MSI channels.
- Registered: `Y_H = P0(X)`, `Y_M = R0(X)`, reconstruct `X`.
- Mixed: exact identity **10%**; deformed-HSI observation **90%**,
  `Y_H = P0(W_phi(X))`; MSI remains in the reference frame.
- Geometry: translation ±4 HR pixels, rotation ±2 degrees, local B-spline
  deformation up to 4 HR pixels, 5x5 control grid, minimum Jacobian 0.5.
- Training budgets from root protocol: 800 registered, 600 mixed.
- Validation monitor: `reg_PSNR` (registered), `warp_PSNR` (mixed).
- Formal mixed test: common **10 geometry seeds x 10 cases** per held-out patch.
- Batch size 1 under the unified launcher; per-method AdamW settings remain
  identical to the official SA-RFM training script.

Supported synthetic x4 datasets are PaviaU, Houston13, Chikusei, CAVE,
Botswana and Augsburg. Real Augsburg/Wald x3 is **not** implemented.

## Run from repository root

First check the official core in a small tensor-only smoke test:

```bash
python comparison/SA-RFM/smoke_test.py --device cpu
# or run on GPU and include full-image 128x128 inference:
python comparison/SA-RFM/smoke_test.py --device cuda:0 --full
```

Inspect the unified command before launching:

```bash
python train.py --model SA-RFM --dataset PaviaU --mode registered --device cuda:0 --dry_run
```

Train registered stage, then mixed:

```bash
python train.py --model SA-RFM --dataset PaviaU --mode registered --device cuda:0
python train.py --model SA-RFM --dataset PaviaU --mode mixed --device cuda:0
```

Resume with a saved optimizer/RNG state:

```bash
python train.py --model SA-RFM --dataset PaviaU --mode registered --device cuda:0 --resume auto
```

Registered final test:

```bash
python test.py --model SA-RFM --dataset PaviaU --mode registered --device cuda:0
```

Mixed main-table test (100 synthetic cases per held-out patch):

```bash
python test.py --model SA-RFM --dataset PaviaU --mode mixed --device cuda:0
```

One-sample, one-seed mixed quick check after training:

```bash
python test.py --model SA-RFM --dataset PaviaU --mode mixed --device cuda:0 \
  --geometry_seeds 10001 --cases 1 --max_samples 1
```

## Outputs

```text
comparison/SA-RFM/checkpoints/physical/<Dataset>/{best,last}.pth.tar
comparison/SA-RFM/checkpoints/hsi_warp_final/<Dataset>/{best,last}.pth.tar
comparison/SA-RFM/logs/{physical,hsi_warp_final}/<Dataset>/validation_history.csv
comparison/SA-RFM/outputs/unified_two_stage/<Dataset>/
comparison/SA-RFM/outputs/multiseed_deformation/<Dataset>/
```

Each checkpoint records architecture ID, network dimensions, Euler steps,
sensor-SRF fingerprint, stage, optimizer state, RNG state, and experiment
arguments. Stage-2 initialization requires the stage-1 best checkpoint.

**Implementation status:** GitHub integration and static code review completed;
there has been no dataset-backed GPU training/validation in this environment.
Do not report any PSNR/SAM before running the smoke test and benchmark.
