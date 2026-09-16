#!/usr/bin/env bash
set -euo pipefail

# Final paper-style UAFL test matched to S2Diff-MH.
# Same trained checkpoint is evaluated under Registered and Warp conditions.
# Warp = dx,dy independently [-4,+4] HR px + rotation [-2,+2] deg +
# local max 4 HR px (5x5 controls -> cubic B-spline), 10 deterministic cases.
# Test patch: center 128x128; full-frame metrics; RMSE table value = raw RMSE*255.

python comparison/UAFL/test_hsi_deformed.py \
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
  --cases 10 \
  --seed 10 \
  --checkpoint comparison/UAFL/checkpoints/hsi_warp_final/PaviaU/best.pth.tar \
  --output_json comparison/UAFL/outputs/uafl_hsi_warp_final_PaviaU_seed10_cases10.json \
  "$@"
