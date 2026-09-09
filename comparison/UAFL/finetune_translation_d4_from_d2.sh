#!/usr/bin/env bash
set -euo pipefail

# Curriculum UAFL non-registration fine-tuning: context d=2 -> context d=4.
# Translation severity uses the shared Euclidean-radius definition:
#   r~U(0,4), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta), |shift|<=4 px.
# HR-MSI is warped on a larger parent then center-cropped to the 64x64 target.
# For d=4 the context launcher uses margin=ceil(d)+2=6 px (76x76 parent).

D2_BEST="comparison/UAFL/checkpoints/physical_translation_d2_context/PaviaU/best.pth.tar"

if [[ ! -f "${D2_BEST}" ]]; then
  echo "Missing context d=2 best checkpoint: ${D2_BEST}" >&2
  exit 1
fi

python comparison/UAFL/train_context_misalignment.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 4.0 \
  --epochs 1200 \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --early_stop_patience 999999 \
  --checkpoint_dir comparison/UAFL/checkpoints/physical_translation_d4_context_ft_d2/PaviaU \
  --log_dir comparison/UAFL/logs/physical_translation_d4_context_ft_d2/PaviaU \
  --resume "${D2_BEST}" \
  "$@"
