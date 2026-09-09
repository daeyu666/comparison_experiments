#!/usr/bin/env bash
set -euo pipefail

# Curriculum UAFL non-registration fine-tuning: strongest radial d=2 -> radial d=4.
# Train patch remains 64x64; only HR-MSI is warped.
# d=4 means sqrt(dx^2+dy^2) <= 4 px:
#   r~U(0,4), theta~U(0,2pi), dx=r*cos(theta), dy=r*sin(theta).
#
# Important: d=2 may have been continued in another checkpoint directory.
# Do NOT assume physical_translation_d2/PaviaU/best.pth.tar is the newest best.
# This launcher scans all non-context UAFL best checkpoints, keeps only
# PaviaU/physical/translation/d=2 checkpoints, selects the one with the largest
# stored best_psnr, preserves model + AdamW + epoch, resets only best_psnr for
# the harder d=4 validation distribution, and trains 400 more epochs.

D4_DIR="comparison/UAFL/checkpoints/physical_translation_d4_ft_strongest_d2/PaviaU"
D4_LOG="comparison/UAFL/logs/physical_translation_d4_ft_strongest_d2/PaviaU"
D4_INIT="${D4_DIR}/resume_strongest_d2_reset_best.pth.tar"
META_FILE="${D4_DIR}/selected_d2_meta.txt"

mkdir -p "${D4_DIR}" "${D4_LOG}"

python - "${D4_INIT}" "${META_FILE}" <<'PY'
from pathlib import Path
import math
import sys
import torch

out_ckpt = Path(sys.argv[1])
meta_file = Path(sys.argv[2])
root = Path("comparison/UAFL/checkpoints")

candidates = []
for path in root.rglob("best.pth.tar"):
    # The warp-before-crop experiment was rejected; never use it as curriculum source.
    if "context" in str(path).lower():
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
    if not math.isfinite(d) or abs(d - 2.0) > 1e-8:
        continue
    if not math.isfinite(best):
        continue
    candidates.append((best, epoch, path, ckpt))

if not candidates:
    raise FileNotFoundError(
        "No non-context UAFL PaviaU physical translation d=2 best checkpoint found under "
        "comparison/UAFL/checkpoints"
    )

candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
best_psnr, source_epoch, source_path, ckpt = candidates[0]

print("Eligible d=2 best checkpoints:")
for psnr, epoch, path, _ in candidates:
    print(f"  PSNR={psnr:.6f} epoch={epoch} path={path}")
print(
    f"Selected strongest d=2 source: PSNR={best_psnr:.6f}, "
    f"epoch={source_epoch}, path={source_path}"
)

ckpt["best_psnr"] = float("-inf")
out_ckpt.parent.mkdir(parents=True, exist_ok=True)
torch.save(ckpt, out_ckpt)

target_epoch = source_epoch + 400
meta_file.write_text(
    f"source_path={source_path}\n"
    f"source_epoch={source_epoch}\n"
    f"source_best_psnr={best_psnr:.10f}\n"
    f"target_epoch={target_epoch}\n",
    encoding="utf-8",
)
print(f"Prepared d=4 resume -> {out_ckpt}; target_epoch={target_epoch}")
PY

TARGET_EPOCHS="$(awk -F= '/^target_epoch=/{print $2}' "${META_FILE}")"
if [[ -z "${TARGET_EPOCHS}" ]]; then
  echo "Failed to resolve target_epoch from ${META_FILE}" >&2
  exit 1
fi

python comparison/UAFL/train.py \
  --dataset PaviaU \
  --degradation_mode physical \
  --train_misalignment_mode translation \
  --translation_max_px 4.0 \
  --epochs "${TARGET_EPOCHS}" \
  --batch_size 1 \
  --lr 1e-5 \
  --weight_decay 5e-5 \
  --early_stop_patience 999999 \
  --checkpoint_dir "${D4_DIR}" \
  --log_dir "${D4_LOG}" \
  --resume "${D4_INIT}" \
  "$@"
