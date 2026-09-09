#!/usr/bin/env bash
set -euo pipefail

# Curriculum UAFL non-registration fine-tuning: radial d=2 -> radial d=4.
# Train patch remains 64x64; only HR-MSI is warped.
# d=4 means sqrt(dx^2+dy^2) <= 4 px:
#   r~U(0,4), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta).
# Resume model + AdamW + epoch from the d=2 BEST checkpoint, but reset only
# best_psnr because validation now uses the harder d=4 distribution.

D2_BEST="comparison/UAFL/checkpoints/physical_translation_d2/PaviaU/best.pth.tar"
D4_DIR="comparison/UAFL/checkpoints/physical_translation_d4_ft_d2/PaviaU"
D4_LOG="comparison/UAFL/logs/physical_translation_d4_ft_d2/PaviaU"
D4_INIT="${D4_DIR}/resume_d2_reset_best.pth.tar"

if [[ ! -f "${D2_BEST}" ]]; then
  echo "Missing d=2 best checkpoint: ${D2_BEST}" >&2
  exit 1
fi

mkdir -p "${D4_DIR}" "${D4_LOG}"
python - "${D2_BEST}" "${D4_INIT}" <<'PY'
import sys
import torch
src, dst = sys.argv[1], sys.argv[2]
ckpt = torch.load(src, map_location="cpu")
if not isinstance(ckpt, dict):
    raise TypeError("Expected a UAFL checkpoint dictionary")
old = ckpt.get("best_psnr", None)
ckpt["best_psnr"] = float("-inf")
torch.save(ckpt, dst)
print(f"Prepared d=4 curriculum resume: {src} -> {dst}; old best_psnr={old}, new=-inf")
PY

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
  --checkpoint_dir "${D4_DIR}" \
  --log_dir "${D4_LOG}" \
  --resume "${D4_INIT}" \
  "$@"
