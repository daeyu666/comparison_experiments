#!/usr/bin/env bash
set -euo pipefail

# Recommended first non-registration run after switching to radial severity.
# GT-HSI and LR-HSI stay fixed; only HR-MSI is translated.
# d=2 means sqrt(dx^2+dy^2) <= 2 px for every sample:
#   r~U(0,2), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta).
# Random initialization; no registered warm-start.
# Keep outputs separate from the earlier d=6 experiment.

python comparison/UAFL/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 2.0 \
  --epochs 800 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --early_stop_patience 999999 \
  --checkpoint_dir comparison/UAFL/checkpoints/physical_translation_d2/PaviaU \
  --log_dir comparison/UAFL/logs/physical_translation_d2/PaviaU \
  "$@"
