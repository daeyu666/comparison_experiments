#!/usr/bin/env bash
set -euo pipefail

# Formal UAFL non-registration training under the shared protocol.
# GT-HSI and LR-HSI stay fixed; only HR-MSI is translated.
# Translation is sampled independently per training sample as dx,dy ~ U(-6,6) px.
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
