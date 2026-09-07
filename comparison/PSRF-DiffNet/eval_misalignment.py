"""Paired translation-misalignment sweep for the PSRF-DiffNet comparison."""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import build_shared_cfg, calc_masked_psnr_sam, resolve_device, set_seed  # noqa: E402
from data_loader import build_datasets  # noqa: E402
from degradations import make_misaligned_msi  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from diffusion import PSRFDiffusion  # noqa: E402


DEFAULT_SHIFTS = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0, 6.0]


def parse_args():
    p = argparse.ArgumentParser(description="PSRF-DiffNet valid-overlap misalignment sweep")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--dataset", default="PaviaU", choices=["PaviaU", "Houston13", "Chikusei"])
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--scale_ratio", type=int, default=4)

    p.add_argument("--degradation_mode", default="physical", choices=["physical", "gaussian_bicubic"])
    p.add_argument("--degradation_sigma", type=float, default=2.0)
    p.add_argument("--degradation_kernel_size", type=int, default=5)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--msi_mode", default="srf", choices=["srf", "uniform"])
    p.add_argument("--srf_path", default="")
    p.add_argument("--wavelength_root", default="./data/wavelengths")
    p.add_argument("--wavelength_path", default="")
    p.add_argument("--srf_interp", default="pchip", choices=["pchip", "linear"])
    p.add_argument(
        "--srf_band_set",
        default="auto",
        choices=["auto", "ikonos4", "wv2_visible5", "wv2_visible6", "wv2_all8"],
    )

    p.add_argument("--misalignment_shifts", type=float, nargs="+", default=DEFAULT_SHIFTS)
    p.add_argument("--misalignment_trials", type=int, default=5)
    p.add_argument("--misalignment_seed", type=int, default=None)
    p.add_argument("--valid_threshold", type=float, default=0.999)
    p.add_argument("--sample_steps", type=int, default=200)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--output", default="")
    return p.parse_args()


def write_csv(path, rows):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    return float(np.mean([float(r[key]) for r in rows]))


@torch.no_grad()
def main():
    args = parse_args()
    if any(float(v) < 0.0 for v in args.misalignment_shifts):
        raise ValueError("misalignment shifts must be >= 0")
    if args.misalignment_trials < 1:
        raise ValueError("misalignment_trials must be >= 1")

    set_seed(args.seed)
    device = resolve_device(args.device)
    ckpt = torch.load(args.checkpoint, map_location=device)
    saved = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}

    args.patch_size = int(saved.get("patch_size", args.patch_size))
    args.image_size = int(saved.get("image_size", args.image_size))
    args.scale_ratio = int(saved.get("scale_ratio", args.scale_ratio))

    cfg = build_shared_cfg(args)
    _, _, test_set, info = build_datasets(cfg, include_validation=True)
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=0)

    model = PSRFDiffusion(
        patch_size=args.patch_size,
        scale_ratio=args.scale_ratio,
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        n_timestep=int(saved.get("n_timestep", 2000)),
        linear_start=float(saved.get("linear_start", 1e-4)),
        linear_end=float(saved.get("linear_end", 2e-3)),
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    print(
        f"Loaded {args.checkpoint}; degradation={args.degradation_mode}, "
        f"HSI={info['n_bands']}, MSI={info['n_select_bands']}, sample_steps={args.sample_steps}"
    )

    cached = list(test_loader)
    base_seed = args.seed if args.misalignment_seed is None else int(args.misalignment_seed)
    details = []

    for shift in [float(v) for v in args.misalignment_shifts]:
        trials = 1 if abs(shift) < 1e-12 else args.misalignment_trials
        for trial in range(trials):
            warp_gen = torch.Generator(device="cpu").manual_seed(base_seed + trial * 100003)
            sample_gen = torch.Generator(device="cpu").manual_seed(base_seed + trial * 100003 + 49999)
            for sample_idx, batch in enumerate(cached):
                gt = batch["gt"].to(device, non_blocking=True)
                lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
                hr_msi = batch["hr_msi"].to(device, non_blocking=True)
                warped, valid, params = make_misaligned_msi(
                    hr_msi,
                    translation_max_px=shift,
                    rotation_max_deg=0.0,
                    local_max_displacement_px=0.0,
                    control_grid_size=5,
                    generator=warp_gen,
                )
                pred, diag = model.sample_tiled(
                    lr_hsi,
                    warped,
                    sample_steps=args.sample_steps,
                    tile_stride=args.patch_size,
                    generator=sample_gen,
                )
                full = calc_metrics(pred, gt, args.scale_ratio)
                psnr_v, sam_v, frac = calc_masked_psnr_sam(
                    pred, gt, valid, threshold=args.valid_threshold
                )
                dx = float(params.dx_px[0].item())
                dy = float(params.dy_px[0].item())
                fine_mean = diag.get("fine_offset_mean_px")
                if fine_mean is None:
                    fine = diag["fine_offset_px"]
                    fine_mean = torch.linalg.vector_norm(fine, dim=-1).mean()
                details.append(
                    {
                        "max_shift_px": shift,
                        "trial": trial,
                        "sample": sample_idx,
                        "dx_px": dx,
                        "dy_px": dy,
                        "actual_shift_px": math.hypot(dx, dy),
                        "valid_fraction": frac,
                        "PSNR_valid": psnr_v,
                        "SAM_valid": sam_v,
                        "PSNR_full": float(full["PSNR"]),
                        "SAM_full": float(full["SAM"]),
                        "RMSE_full": float(full["RMSE"]),
                        "ERGAS_full": float(full["ERGAS"]),
                        "SSIM_full": float(full["SSIM"]),
                        "CC_full": float(full["CC"]),
                        "fine_offset_mean_px": float(fine_mean.item()),
                    }
                )

    groups = defaultdict(list)
    for row in details:
        groups[float(row["max_shift_px"])].append(row)

    summary = []
    for shift in [float(v) for v in args.misalignment_shifts]:
        rows = groups[shift]
        summary.append(
            {
                "max_shift_px": shift,
                "n_runs": len(rows),
                "mean_actual_shift_px": mean(rows, "actual_shift_px"),
                "valid_fraction": mean(rows, "valid_fraction"),
                "PSNR_valid": mean(rows, "PSNR_valid"),
                "SAM_valid": mean(rows, "SAM_valid"),
                "PSNR_full": mean(rows, "PSNR_full"),
                "SAM_full": mean(rows, "SAM_full"),
                "RMSE_full": mean(rows, "RMSE_full"),
                "ERGAS_full": mean(rows, "ERGAS_full"),
                "SSIM_full": mean(rows, "SSIM_full"),
                "CC_full": mean(rows, "CC_full"),
                "fine_offset_mean_px": mean(rows, "fine_offset_mean_px"),
            }
        )

    if not args.output:
        args.output = str(THIS_DIR / "outputs" / args.degradation_mode / args.dataset / "translation_sweep.csv")
    write_csv(args.output, summary)
    stem, ext = os.path.splitext(args.output)
    detail_path = f"{stem}_details{ext or '.csv'}"
    write_csv(detail_path, details)

    print("\n=== PSRF-DiffNet translation sensitivity (valid overlap) ===")
    for row in summary:
        print(
            f"{row['max_shift_px']:>4.1f}px  actual={row['mean_actual_shift_px']:.3f}px  "
            f"PSNR={row['PSNR_valid']:.4f}  SAM={row['SAM_valid']:.4f}  "
            f"valid={row['valid_fraction']:.4f}"
        )
    print(f"Summary CSV: {args.output}")
    print(f"Details CSV: {detail_path}")


if __name__ == "__main__":
    main()
