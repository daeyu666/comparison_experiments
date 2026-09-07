"""Evaluate HSIFN under the shared unregistered HSI-MSI protocol.

Two protocols are provided:
  * translation_sweep: paired 0/0.5/1/2/3/4/6 px sensitivity curve;
  * full: registered / translation / rotation / global / local / global_local.

Metrics use the same valid-overlap definition as S2Diff.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    build_shared_cfg,
    calc_masked_psnr_sam,
    require_srf_weights,
    resolve_device,
    set_seed,
    warp_msi_for_protocol,
)
from data_loader import build_datasets  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from model import HSIFN  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="HSIFN misalignment evaluation")
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

    p.add_argument("--checkpoint", default="")
    p.add_argument("--split", default="validation", choices=["validation", "test"])
    p.add_argument("--protocol", default="translation_sweep", choices=["translation_sweep", "full"])
    p.add_argument("--shifts", type=float, nargs="+", default=[0, 0.5, 1, 2, 3, 4, 6])
    p.add_argument("--trials", type=int, default=5)
    p.add_argument("--rotation_max_deg", type=float, default=3.0)
    p.add_argument("--local_max_displacement_px", type=float, default=3.0)
    p.add_argument("--translation_max_px", type=float, default=6.0)
    p.add_argument("--control_grid_size", type=int, default=5)
    p.add_argument("--valid_threshold", type=float, default=0.999)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument("--output", default="")
    return p.parse_args()


def _load_model(args, info, device):
    ckpt_path = args.checkpoint or str(
        THIS_DIR / "checkpoints" / args.degradation_mode / args.dataset / "best.pth.tar"
    )
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) if isinstance(ckpt, dict) else {}
    use_mask = not bool(ckpt_args.get("no_mask", False))
    model = HSIFN(
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        srf_weights=require_srf_weights(info),
        use_mask=use_mask,
    ).to(device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=True)
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path}")
    return model


@torch.no_grad()
def _run_once(model, batch, device, args, *, mode, tmax, rmax, lmax, seed) -> Dict[str, float]:
    gt = batch["gt"].to(device)
    lr_hsi = batch["lr_hsi"].to(device)
    hr_msi = batch["hr_msi"].to(device)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    warped_msi, valid, mis = warp_msi_for_protocol(
        hr_msi,
        mode=mode,
        translation_max_px=float(tmax),
        rotation_max_deg=float(rmax),
        local_max_displacement_px=float(lmax),
        control_grid_size=args.control_grid_size,
        generator=generator,
    )
    pred, diag = model(lr_hsi, warped_msi)
    pred = pred.clamp(0.0, 1.0)
    full = calc_metrics(pred, gt, args.scale_ratio)
    psnr_valid, sam_valid, valid_fraction = calc_masked_psnr_sam(
        pred, gt, valid, threshold=args.valid_threshold
    )
    actual_shift = torch.sqrt(mis.dx_px.square() + mis.dy_px.square()).mean().item()
    flow_mean = torch.linalg.vector_norm(diag["coarse_flow"], dim=1).mean().item()
    mask_mean = (
        torch.stack([m.mean() for m in diag["masks"]]).mean().item()
        if diag["masks"] else float("nan")
    )
    return {
        **{k: float(v) for k, v in full.items()},
        "PSNR_valid": float(psnr_valid),
        "SAM_valid": float(sam_valid),
        "valid_fraction": float(valid_fraction),
        "actual_translation_px": float(actual_shift),
        "predicted_flow_mean_px": float(flow_mean),
        "mask_mean": float(mask_mean),
        "dx_px": float(mis.dx_px.mean().item()),
        "dy_px": float(mis.dy_px.mean().item()),
        "rotation_deg": float(mis.rotation_deg.mean().item()),
        "local_displacement_px": float(mis.local_displacement_px.mean().item()),
    }


def _aggregate(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = rows[0].keys()
    return {k: sum(float(r[k]) for r in rows) / len(rows) for k in keys}


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = build_shared_cfg(args)
    _, val_set, test_set, info = build_datasets(cfg, include_validation=True)
    dataset = val_set if args.split == "validation" else test_set
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    batch = next(iter(loader))
    device = resolve_device(args.device)
    model = _load_model(args, info, device)

    details: List[Dict] = []
    summary: List[Dict] = []

    if args.protocol == "translation_sweep":
        for severity in args.shifts:
            trial_rows = []
            n_trials = 1 if float(severity) == 0.0 else args.trials
            for trial in range(n_trials):
                # Resetting the generator to the same trial seed for each severity
                # gives paired random directions; dx/d scale linearly with severity.
                row = _run_once(
                    model, batch, device, args,
                    mode="registered" if float(severity) == 0.0 else "translation",
                    tmax=float(severity), rmax=0.0, lmax=0.0,
                    seed=args.seed + 1009 * trial,
                )
                row.update({"scenario": "translation", "severity": float(severity), "trial": trial})
                details.append(row)
                trial_rows.append(row)
            agg = _aggregate([
                {k: v for k, v in r.items() if isinstance(v, (float, int)) and k not in ("severity", "trial")}
                for r in trial_rows
            ])
            agg.update({"scenario": "translation", "severity": float(severity), "trials": n_trials})
            summary.append(agg)
            print(
                f"shift={severity:g}px actual={agg['actual_translation_px']:.3f}px "
                f"PSNR_valid={agg['PSNR_valid']:.4f} SAM_valid={agg['SAM_valid']:.4f} "
                f"flow={agg['predicted_flow_mean_px']:.3f}px"
            )
    else:
        scenarios = [
            ("registered", 0.0, 0.0, 0.0),
            ("translation", args.translation_max_px, 0.0, 0.0),
            ("rotation", 0.0, args.rotation_max_deg, 0.0),
            ("global", args.translation_max_px, args.rotation_max_deg, 0.0),
            ("local", 0.0, 0.0, args.local_max_displacement_px),
            ("global_local", args.translation_max_px, args.rotation_max_deg, args.local_max_displacement_px),
        ]
        for name, tmax, rmax, lmax in scenarios:
            trial_rows = []
            n_trials = 1 if name == "registered" else args.trials
            for trial in range(n_trials):
                row = _run_once(
                    model, batch, device, args,
                    mode=name, tmax=tmax, rmax=rmax, lmax=lmax,
                    seed=args.seed + 1009 * trial,
                )
                row.update({"scenario": name, "trial": trial})
                details.append(row)
                trial_rows.append(row)
            numeric_rows = [
                {k: v for k, v in r.items() if isinstance(v, (float, int)) and k != "trial"}
                for r in trial_rows
            ]
            agg = _aggregate(numeric_rows)
            agg.update({"scenario": name, "trials": n_trials})
            summary.append(agg)
            print(
                f"{name:12s} PSNR_valid={agg['PSNR_valid']:.4f} "
                f"SAM_valid={agg['SAM_valid']:.4f} flow={agg['predicted_flow_mean_px']:.3f}px"
            )

    if not args.output:
        suffix = "translation_sweep" if args.protocol == "translation_sweep" else "full_misalignment"
        args.output = str(
            THIS_DIR / "outputs" / args.degradation_mode / args.dataset / f"{suffix}.csv"
        )
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    def write_csv(path, rows):
        if not rows:
            return
        fields = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output, summary)
    detail_path = os.path.splitext(args.output)[0] + "_details.csv"
    write_csv(detail_path, details)
    print(f"Saved summary: {args.output}")
    print(f"Saved details: {detail_path}")


if __name__ == "__main__":
    main()
