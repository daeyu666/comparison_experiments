#!/usr/bin/env bash
set -euo pipefail

# UAFL non-registration training under the shared radial translation protocol.
# GT-HSI and LR-HSI stay fixed; only HR-MSI is translated.
# IMPORTANT: d=6 means total Euclidean displacement <= 6 px:
#   r~U(0,6), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta).
# This run starts from random initialization (no registered checkpoint warm-start).
# Validation remains every 20 epochs for PaviaU and best.pth.tar is still updated.

python comparison/UAFL/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 6.0 \
  --epochs 800 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --early_stop_patience 999999 \
  "$@"
