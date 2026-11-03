# IR&ArF adapter for comparison_experiments

Upstream implementation: https://github.com/Jiahuiqu/IR-ArF  
Paper: J. Qu et al., "IR&ArF: Toward Deep Interpretable Arbitrary Resolution Fusion of Unregistered Hyperspectral and Multispectral Images," IEEE TIP, 2025. DOI: 10.1109/TIP.2025.3551531.

This adapter retains the upstream model blocks and three-stage fusion/denoising
topology. Engineering adaptations:
- Dynamic HSI band count (upstream Pavia setting was fixed at 102).
- Dynamic MSI band count (upstream implementation used 4 MSI channels).
- Read the actual LR-HSI spatial size instead of hard-coding 20 pixels.
- Create coordinate tensors on the active input device rather than a fixed CUDA device.
- Retain the upstream L1 + weighted NCC training objective in the benchmark trainer.

The shared `data_loader.build_datasets` remains authoritative for scene splits,
normalization, spatial degradation, SRF integration and HSI-side deformation.
The upstream fixed-shape dataloader is not reused.

Initial target: PaviaU (103 HSI bands, 4 MSI bands), x4 scale, shared 64x64 HR
patch and 16x16 LR-HSI. Run `python comparison/IR-ArF/smoke_test.py` before
training. The adapter is not yet validated on the remaining datasets.

## Completed integration

- Official three-iteration fusion/denoising architecture and L1 + 0.1 NCC loss.
- Dynamic HSI channels (103/128/144/145/31/242) and MSI channels (3/4/8).
- Shared synthetic x4 SRF / MTF-PSF degradation and HSI-side deformation.
- Shared patch 64, stride 32, image 128; registered/mixed 800/600 epochs.
- Method-specific Adam lr=2e-4 and StepLR(20, gamma=0.8), not AdamW.
- Mixed validation uses deterministic nonregistered deformation cases; best checkpoint
  is selected on warp PSNR, while registered training uses registered PSNR.
- Checkpoint source, sensor fingerprint, architecture, training protocol, optimizer,
  scheduler, and RNG state; formal outputs remain separate from debug runs.
- Shared fixed-deformation suite and 10-seed x 10-case main-table evaluation.

## Environment

The comparison adapter uses standard PyTorch, scipy, numpy, matplotlib and tqdm,
without compiling any IR&ArF-specific CUDA extension. Install dependencies into
the existing comparison environment without changing its PyTorch/CUDA version:

```bash
python -m pip install "numpy<2" scipy matplotlib tqdm
```

## Commands (from repository root)

```bash
git pull origin main

python comparison/IR-ArF/smoke_test.py

# Debug: only one train batch, isolated in checkpoints/physical_smoke
python comparison/IR-ArF/train.py --dataset PaviaU --stage registered \
  --device cuda:0 --epochs 1 --max_train_batches 1

# Full-data one-epoch sanity check; writes the formal registered checkpoint
python train.py --model IR-ArF --dataset PaviaU --mode registered \
  --device cuda:0 --epochs 1

# Continue to the common 800-epoch budget
python train.py --model IR-ArF --dataset PaviaU --mode registered \
  --device cuda:0 --epochs 800 --resume auto

# Stage-2: reads the best registered checkpoint, then trains 600 epochs
python train.py --model IR-ArF --dataset PaviaU --mode mixed --device cuda:0

# Final tests
python test.py --model IR-ArF --dataset PaviaU --mode registered --device cuda:0
python test.py --model IR-ArF --dataset PaviaU --mode mixed --device cuda:0
python test_fixed_deformation.py --model IR-ArF --dataset PaviaU \
  --mode mixed --device cuda:0
```

The mixed main-table test uses the central `test_multiseed_deformation.py`
runner and supports `--max_samples`, `--resume_output`, `--overwrite_output`
and `--geometry_seeds`. Do not compare the single-batch debug weights against
full-budget methods.

## Verification status

Static source-level integration and repository registration have been inspected.
A runtime forward/backward smoke test, full-data one-epoch training run, and
GPU-memory check on the target server remain necessary. Do not treat this
adapter as a validated reproduction until those checks pass.
