#!/usr/bin/env bash
set -euo pipefail

# Formal UAFL comparison matched to S2Diff-MH:
# PaviaU; train HR patch 64x64; validation/test HR patch 128x128.
# HSI observation: X -> W_phi(X) -> physical P0 -> LR-HSI.
# HR-MSI remains registered/reliable.
# Geometry: dx,dy independently in [-4,+4] HR px; rotation in [-2,+2] deg;
# local non-rigid max 4 HR px from a 5x5 zero-mean control grid + cubic B-spline.
# Warm-start model weights from the registered UAFL best; optimizer is reinitialized.

python comparison/UAFL/train_hsi_deformed.py \
  --dataset PaviaU \
  --image_size 128 \
  --patch_size 64 \
  --stride 32 \
  --scale_ratio 4 \
  --degradation_mode physical \
  --mtf_nyquist 0.2 \
  --psf_truncate 3.0 \
  --max_translation 4.0 \
  --max_rotation_deg 2.0 \
  --max_local_px 4.0 \
  --control_grid 5 \
  --min_jacobian 0.5 \
  --epochs 400 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --seed 10 \
  --validation_interval 20 \
  --validation_cases 5 \
  --early_stop_patience 999999 \
  --init_checkpoint comparison/UAFL/checkpoints/physical/PaviaU/best.pth.tar \
  --checkpoint_dir comparison/UAFL/checkpoints/hsi_warp_final/PaviaU \
  --log_dir comparison/UAFL/logs/hsi_warp_final/PaviaU \
  "$@"
