#!/usr/bin/env bash
set -euo pipefail

# Formal UAFL d=2 non-registration run using warp-before-crop geometry.
# GT-HSI/LR-HSI target stays 64x64. HR-MSI is generated on a larger parent,
# translated there, then center-cropped back to 64x64.
# d=2 means sqrt(dx^2+dy^2) <= 2 px:
#   r~U(0,2), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta).
# The context launcher uses margin=ceil(d)+2=4 px, i.e. a 72x72 MSI parent.
# Random initialization; no registered warm-start.

python comparison/UAFL/train_context_misalignment.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 2.0 \
  --epochs 800 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --early_stop_patience 999999 \
  --checkpoint_dir comparison/UAFL/checkpoints/physical_translation_d2_context/PaviaU \
  --log_dir comparison/UAFL/logs/physical_translation_d2_context/PaviaU \
  "$@"
