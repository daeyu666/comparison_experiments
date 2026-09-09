#!/usr/bin/env bash
set -euo pipefail

# Rotation curriculum stage 1:
# strongest radial translation d=6 -> global translation d=6 + rotation <= 1 deg.
# Train patch remains 64x64; only HR-MSI is warped.
# Local non-rigid deformation stays disabled.
#
# The launcher scans all non-context UAFL best checkpoints, keeps only
# PaviaU/physical/translation/d=6 checkpoints, selects the one with the largest
# stored best_psnr, preserves model + AdamW + epoch, resets only best_psnr for
# the new rotation stage, and trains 180 more epochs.

ROT1_DIR="comparison/UAFL/checkpoints/physical_global_d6_rot1_ft_strongest_d6/PaviaU"
ROT1_LOG="comparison/UAFL/logs/physical_global_d6_rot1_ft_strongest_d6/PaviaU"
ROT1_INIT="${ROT1_DIR}/resume_strongest_d6_reset_best.pth.tar"
META_FILE="${ROT1_DIR}/selected_d6_meta.txt"

mkdir -p "${ROT1_DIR}" "${ROT1_LOG}"

python - "${ROT1_INIT}" "${META_FILE}" <<'PY'
from pathlib import Path
import math
import sys
import torch

out_ckpt = Path(sys.argv[1])
meta_file = Path(sys.argv[2])
root = Path("comparison/UAFL/checkpoints")

candidates = []
for path in root.rglob("best.pth.tar"):
    text = str(path).lower()
    if "context" in text:
        continue
    try:
        ckpt = torch.load(path, map_location="cpu")
    except Exception as exc:
        print(f"skip unreadable checkpoint {path}: {exc}")
        continue
    if not isinstance(ckpt, dict):
        continue
    args = ckpt.get("args", {}) or {}
    if args.get("dataset") != "PaviaU":
        continue
    if args.get("degradation_mode") != "physical":
        continue
    if args.get("train_misalignment_mode") != "translation":
        continue
    try:
        d = float(args.get("translation_max_px", float("nan")))
        best = float(ckpt.get("best_psnr", float("-inf")))
        epoch = int(ckpt.get("epoch", -1))
    except Exception:
        continue
    if not math.isfinite(d) or abs(d - 6.0) > 1e-8:
        continue
    if not math.isfinite(best):
        continue
    candidates.append((best, epoch, path, ckpt))

if not candidates:
    raise FileNotFoundError(
        "No non-context UAFL PaviaU physical translation d=6 best checkpoint found under "
        "comparison/UAFL/checkpoints"
    )

candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
best_psnr, source_epoch, source_path, ckpt = candidates[0]

print("Eligible d=6 translation best checkpoints:")
for psnr, epoch, path, _ in candidates:
    print(f"  PSNR={psnr:.6f} epoch={epoch} path={path}")
print(
    f"Selected strongest d=6 source: PSNR={best_psnr:.6f}, "
    f"epoch={source_epoch}, path={source_path}"
)

ckpt["best_psnr"] = float("-inf")
out_ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out_ckpt)

target_epoch = source_epoch + 180
meta_file.write_text(
    f"source_path={source_path}\n"
    f"source_epoch={source_epoch}\n"
    f"source_best_psnr={best_psnr:.10f}\n"
    f"target_epoch={target_epoch}\n",
    encoding="utf-8",
)
print(f"Prepared d=6 + rot1 resume -> {out_ckpt}; target_epoch={target_epoch}")
PY

TARGET_EPOCHS="$(awk -F= '/^target_epoch=/{print $2}' "${META_FILE}")"
if [[ -z "${TARGET_EPOCHS}" ]]; then
  echo "Failed to resolve target_epoch from ${META_FILE}" >&2
  exit 1
fi

python comparison/UAFL/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode global \
  --translation_max_px 6.0 \
  --rotation_max_deg 1.0 \
  --local_max_displacement_px 0.0 \
  --epochs "${TARGET_EPOCHS}" \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --validation_interval 20 \
  --early_stop_patience 999999 \
  --checkpoint_dir "${ROT1_DIR}" \
  --log_dir "${ROT1_LOG}" \
  --resume "${ROT1_INIT}" \
  "$@"
