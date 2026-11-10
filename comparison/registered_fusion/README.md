# MoG-DCN and ESFS: registered fusion benchmark

From the repository root, replace `ESFS` with `MoG-DCN` to switch methods:

```bash
python train.py --model ESFS --dataset PaviaU --mode registered --device cuda:0
python test.py  --model ESFS --dataset PaviaU --mode registered --device cuda:0
python train.py --model ESFS --dataset PaviaU --mode registered --device cuda:0 --resume auto
```

The root entries use the common ×4 physical degradation (MTF 0.2), sensor
SRF, train/validation/test splits and metrics. Train on 64×64 HR patches
with stride 32; validate and test on 128×128 patches. The root default is
800 epochs, with the repository's dataset-specific validation interval.
Test uses the maximum validation PSNR checkpoint. Both methods only
support registered evaluation in this benchmark.

For the earlier Gaussian + bicubic protocol, use the method-local train
and test scripts with `--degradation_mode gaussian_bicubic`. The default
checkpoint, log and output folders are separate for each degradation:

```text
comparison/<Method>/checkpoints/<degradation>/<Dataset>/{best,last}.pth.tar
comparison/<Method>/logs/<degradation>/<Dataset>/validation_history.csv
comparison/<Method>/outputs/<degradation>/<Dataset>/summary.json
```

The method-local training script also accepts `--data_root`, custom checkpoint
and log directories, and `--early_stop_patience 10` for ten validation checks
without a PSNR gain exceeding 0.02 dB. When resuming through this script,
pass the path to `last.pth.tar` and use the original training settings.
Checkpoints record the method, channel counts, degradation, SRF fingerprint,
optimizer, scheduler and RNG state.

For a small CPU or CUDA check, write into separate directories:

```bash
python comparison/ESFS/train.py --dataset PaviaU --device cpu --epochs 1 \
  --max_train_batches 1 --checkpoint_dir /tmp/esfs-smoke/ckpt --log_dir /tmp/esfs-smoke/logs
python test.py --model ESFS --dataset PaviaU --mode registered --device cpu \
  --checkpoint /tmp/esfs-smoke/ckpt/best.pth.tar --max_samples 1 --output_dir /tmp/esfs-smoke/test
```

CPU checks on PyTorch 2.5.1 covered matched-weight forward comparisons against
the authors' models, ×4 training with 128/8 bands, ×4 inference with 242/4
bands and a synthetic PaviaU train/resume/test cycle. CUDA and real dataset
metrics must be checked on the training server. Train new weights under the
shared benchmark; smoke outputs are not paper results.
