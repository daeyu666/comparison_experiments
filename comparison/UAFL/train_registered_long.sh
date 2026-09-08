#!/usr/bin/env bash
set -euo pipefail

# Continue the paper-faithful UAFL registered PaviaU run beyond 300 epochs.
# Early stopping is effectively disabled by using a very large patience value;
# validation and best-checkpoint tracking remain active every 20 epochs.
python comparison/UAFL/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode registered \
  --epochs 800 \
  --early_stop_patience 999999 \
  --resume comparison/UAFL/checkpoints/physical/PaviaU/last.pth.tar
