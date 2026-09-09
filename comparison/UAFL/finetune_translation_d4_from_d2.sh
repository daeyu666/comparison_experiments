#!/usr/bin/env bash
set -euo pipefail

# Curriculum UAFL non-registration fine-tuning: radial d=2 -> radial d=4.
# Translation severity uses the shared Euclidean-radius definition:
#   r~U(0,4), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta), |shift|<=4 px.
# GT-HSI and LR-HSI remain fixed; only HR-MSI is translated.
# Resume the d=2 BEST checkpoint, including AdamW state, and continue training.
# Outputs are isolated from both the d=2 run and the earlier failed legacy d=6 run.

D2_BEST="comparison/UAFL/checkpoints/physical_translation_d2/PaviaU/best.pth.tar"

if [[ ! -f "${D2_BEST}" ]]; then
  echo "Missing d=2 best checkpoint: ${D2_BEST}" >&2
  exit 1
fi

python comparison/UAFL/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 4.0 \
  --epochs 1200 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --early_stop_patience 999999 \
  --checkpoint_dir comparison/UAFL/checkpoints/physical_translation_d4_ft_d2/PaviaU \
  --log_dir comparison/UAFL/logs/physical_translation_d4_ft_d2/PaviaU \
  --resume "${D2_BEST}" \
  "$@"
