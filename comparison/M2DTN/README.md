# M2DTN comparison adapter

Official implementation: [Jiahuiqu/M2DTN](https://github.com/Jiahuiqu/M2DTN), pinned to
`e50295ae03aeaf902732b3b05d8e527e8130aae1`.

Paper: Wenqian Dong, Yang Xu, Jiahui Qu, Shaoxiong Hou, **Learning Multi-Modal
Cross-Scale Deformable Transformer Network for Unregistered Hyperspectral Image
Super-resolution**, AAAI 2024, DOI 10.1609/aaai.v38i2.27923.

This adapter retains the official M2DTN path: four-scale HSI/MSI transformer
features, MMRD registration, cross-scale feature aggregation and L1 reconstruction.

## Shared benchmark protocol

The formal comparison follows the same settings as the other methods in this repository:

- synthetic x4 datasets: PaviaU / Houston13 / Chikusei / CAVE / Botswana / Augsburg;
- shared sensor SRF for HR-MSI;
- shared physical LR-HSI degradation: MTF@Nyquist=0.2, PSF truncate=3.0;
- train patch 64, stride 32, validation/test patch 128;
- stage 1 registered: 800 epochs, seed 1001;
- stage 2 mixed: 600 epochs, seed 10, 10% exact identity + 90% HSI deformation;
- geometry: dx/dy U(-4,+4) HR px, rotation U(-2,+2) degrees, local <=4 HR px,
  5x5 B-spline controls, min Jacobian 0.5;
- AdamW, lr=1e-5, weight decay=5e-5, batch=1;
- official M2DTN L1 reconstruction loss;
- dataset-aware validation interval inherited from `experiment_protocol.py`;
- mixed main-table test uses the shared 10 geometry seeds x 10 cases protocol.

The AAAI paper reports Adam with learning rate 0.001 and four scale layers. The benchmark
deliberately keeps the repository-wide optimizer/training protocol instead of reproducing
that paper-specific optimization schedule, so this is a controlled comparison reproduction.

## Necessary compatibility changes

The upstream source hard-codes the spatial transformer to `cuda:0`; this adapter makes
resampling use the runtime tensor device without changing the deformation equations.

The upstream `MSVT` defaults to 4 attention heads, while the released code also instantiates
102-band Pavia and demonstrates 31-band input, neither of which is divisible by 4. The adapter
therefore uses the largest valid divisor not exceeding 4 for each modality while leaving the
feature width and q/k/v projections unchanged. This avoids adding external feature projection
layers. The selected head counts are saved in every checkpoint.

M2DTN's four-scale topology is intrinsically x4. Synthetic Augsburg x4 is supported; the
Augsburg-2 real-world Wald x3 path is not exposed because changing it to x3 would alter the
published cross-scale architecture.

## Commands

From repository root:

```bash
python comparison/M2DTN/smoke_test.py

python train.py --model M2DTN --dataset PaviaU --mode registered
python train.py --model M2DTN --dataset PaviaU --mode mixed

python test.py --model M2DTN --dataset PaviaU --mode registered
python test.py --model M2DTN --dataset PaviaU --mode mixed
python test_fixed_deformation.py --model M2DTN --dataset PaviaU --mode mixed
```

Resume training with the standard root interface:

```bash
python train.py --model M2DTN --dataset PaviaU --mode registered --epochs 1200 --resume auto
```

Outputs remain isolated under `comparison/M2DTN/checkpoints`, `logs`, and `outputs`.
