#!/usr/bin/env bash
set -euo pipefail

# Formal UAFL d=2 non-registration run under the shared radial translation protocol.
# Train patch remains 64x64; only the already-generated 64x64 HR-MSI is warped.
# GT-HSI and LR-HSI stay fixed. d is the Euclidean displacement-radius bound:
#   r~U(0,2), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta), |shift|<=2 px.
# Random initialization; no registered warm-start.

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
