"""Final UAFL test matched to the S2Diff-MH deformed-HSI protocol.

Reports two rows from the SAME trained UAFL checkpoint:
  Registered: Y_H = P0(X)
  Warp:       Y_H = P0(W_phi(X)), averaged over deterministic synthetic cases

The HR-MSI stays registered in both rows.  Test geometry and case seed match the
S2Diff-MH final-test convention.  Metrics use the same formulas; RMSE is stored
raw in [0,1] and additionally reported as RMSE*255 to match the paper table.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from statistics import mean

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import build_shared_cfg, require_srf_weights, resolve_device, set_seed  # noqa: E402
from data_loader import build_datasets  # noqa: E402
from hsi_deformation import jacobian_determinant, make_deformed_lr_hsi, sample_synthetic_geometry  # noqa: E402
from model import build_uafl  # noqa: E402


METRIC_ORDER = ("PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE")


def parse_args():
    p = argparse.ArgumentParser(description="Final UAFL deformed-HSI full-metric test")
    p.add_argument("--dataset", default="PaviaU", choices=["PaviaU"])
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--degradation_mode", default="physical", choices=["physical"])
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--msi_mode", default="srf", choices=["srf"])
    p.add_argument("--srf_path", default="")
    p.add_argument("--wavelength_root", default="./data/wavelengths")
    p.add_argument("--wavelength_path", default="")
    p.add_argument("--srf_interp", default="pchip", choices=["pchip", "linear"])
    p.add_argument("--srf_band_set", default="auto")

    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--checkpoint", default="comparison/UAFL/checkpoints/hsi_warp_final/PaviaU/best.pth.tar")
    p.add_argument("--print_cases", action="store_true")
    p.add_argument("--output_json", default="comparison/UAFL/outputs/uafl_hsi_warp_final_PaviaU_seed10_cases10.json")
    return p.parse_args()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))
    return gen


def upsample_lr_hsi(lr_hsi: torch.Tensor, hr_size) -> torch.Tensor:
    return F.interpolate(lr_hsi, size=tuple(hr_size), mode="bicubic", align_corners=False)


# Exact metric formulas used by S2Diff-MH final reporting.
def calc_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred = torch.clamp(pred.detach().float(), 0.0, 1.0)
    target = torch.clamp(target.detach().float(), 0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    return math.sqrt(max(mse, 1e-12))


def calc_psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    rmse = calc_rmse(pred, target)
    return 100.0 if rmse <= 1e-12 else 20.0 * math.log10(1.0 / rmse)


def calc_sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.detach().float()
    target = target.detach().float()
    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.sqrt(torch.sum(pred * pred, dim=1) + eps)
    target_norm = torch.sqrt(torch.sum(target * target, dim=1) + eps)
    cos = dot / (pred_norm * target_norm + eps)
    cos = torch.clamp(cos, -1.0 + eps, 1.0 - eps)
    return torch.mean(torch.acos(cos) * 180.0 / math.pi).item()


def calc_cc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.detach().float().view(pred.shape[0], pred.shape[1], -1)
    target = target.detach().float().view(target.shape[0], target.shape[1], -1)
    pred_centered = pred - pred.mean(dim=2, keepdim=True)
    target_centered = target - target.mean(dim=2, keepdim=True)
    numerator = torch.sum(pred_centered * target_centered, dim=2)
    denominator = torch.sqrt(torch.sum(pred_centered ** 2, dim=2) * torch.sum(target_centered ** 2, dim=2) + eps)
    return torch.mean(numerator / (denominator + eps)).item()


def calc_ergas(pred: torch.Tensor, target: torch.Tensor, scale_ratio: int, eps: float = 1e-8) -> float:
    pred = pred.detach().float()
    target = target.detach().float()
    rmse_per_band = torch.sqrt(torch.mean((pred - target) ** 2, dim=(0, 2, 3)) + eps)
    mean_target = torch.mean(target, dim=(0, 2, 3))
    return (100.0 / scale_ratio * torch.sqrt(torch.mean((rmse_per_band / (mean_target + eps)) ** 2))).item()


def calc_ssim(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred = pred.detach().float()
    target = target.detach().float()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    mu_x, mu_y = pred.mean(), target.mean()
    sigma_x, sigma_y = pred.var(unbiased=False), target.var(unbiased=False)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()
    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim.item()


def calc_metrics(pred: torch.Tensor, target: torch.Tensor, scale_ratio: int):
    rmse = calc_rmse(pred, target)
    return {
        "PSNR": calc_psnr(pred, target),
        "SSIM": calc_ssim(pred, target),
        "ERGAS": calc_ergas(pred, target, scale_ratio),
        "SAM": calc_sam(pred, target),
        "CC": calc_cc(pred, target),
        "RMSE": rmse,
        "RMSE_x255": rmse * 255.0,
    }


def average_metrics(rows):
    keys = rows[0].keys()
    return {k: float(mean(float(r[k]) for r in rows)) for k in keys}


def format_metrics(m):
    return (
        f"PSNR={m['PSNR']:.4f} SSIM={m['SSIM']:.6f} ERGAS={m['ERGAS']:.4f} "
        f"SAM={m['SAM']:.4f} CC={m['CC']:.6f} "
        f"RMSE={m['RMSE_x255']:.4f} (raw={m['RMSE']:.6f})"
    )


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("--cases must be >=1")
    set_seed(args.seed)
    device = resolve_device(args.device)

    cfg = build_shared_cfg(args)
    _train_set, _val_set, test_set, info = build_datasets(cfg, include_validation=True)
    require_srf_weights(info)
    if len(test_set) != 1:
        raise ValueError(f"expected the standard single 128x128 test patch, got {len(test_set)}")
    batch = test_set[0]
    gt = batch["gt"].unsqueeze(0).to(device)
    hr_msi = batch["hr_msi"].unsqueeze(0).to(device)
    h, w = gt.shape[-2:]
    if (h, w) != (args.image_size, args.image_size):
        raise ValueError(f"test patch is {(h, w)}, expected {(args.image_size, args.image_size)}")

    p0 = test_set.degradation_operator.to(device)
    model = build_uafl(int(info["n_select_bands"])).to(device)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"UAFL deformed-HSI checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
    model.eval()

    test_conditions = {
        "dataset": args.dataset,
        "test_patch": f"{h}x{w}",
        "metric_region": "full-frame",
        "cases": args.cases,
        "seed": args.seed,
        "synthetic_case_generator_seed": args.seed + 70000,
        "scale_ratio": args.scale_ratio,
        "physical_degradation": "warp HR-HSI first, then calibrated PSF/MTF + detector integration + x4 sampling",
        "mtf_nyquist": args.mtf_nyquist,
        "psf_truncate": args.psf_truncate,
        "max_translation_hr_px_per_axis": args.max_translation,
        "translation_sampling": f"dx,dy independently U(-{args.max_translation},+{args.max_translation})",
        "max_rotation_deg": args.max_rotation_deg,
        "max_local_hr_px": args.max_local_px,
        "local_deformation": f"{args.control_grid}x{args.control_grid} zero-mean controls -> cubic B-spline dense field",
        "min_synthetic_jacobian": args.min_jacobian,
        "msi_coordinate_system": "reliable HR reference coordinate system; MSI is never warped",
        "hsi_observation": "Y_H=P0(W_phi(X))",
        "metric_implementation": "S2Diff-MH-matched formulas; RMSE table value is raw RMSE*255",
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "checkpoint_best_warp_val_psnr": ckpt.get("best_psnr") if isinstance(ckpt, dict) else None,
    }

    with torch.no_grad():
        registered_lr = p0.degrade(gt)
        registered_pred = model(upsample_lr_hsi(registered_lr, (h, w)), hr_msi)
        registered_metrics = calc_metrics(registered_pred, gt, args.scale_ratio)

    generator = make_generator(device, args.seed + 70000)
    warp_rows = []
    per_case = []
    with torch.no_grad():
        for case_idx in range(args.cases):
            geometry = sample_synthetic_geometry(
                h,
                w,
                device=device,
                dtype=gt.dtype,
                generator=generator,
                max_translation=args.max_translation,
                max_rotation_deg=args.max_rotation_deg,
                max_local_px=args.max_local_px,
                control_grid=args.control_grid,
                min_jacobian=args.min_jacobian,
            )
            warped_lr = make_deformed_lr_hsi(gt, geometry, p0)
            pred = model(upsample_lr_hsi(warped_lr, (h, w)), hr_msi)
            metrics = calc_metrics(pred, gt, args.scale_ratio)
            warp_rows.append(metrics)
            record = {
                "case": case_idx + 1,
                "dx_hr_px": float(geometry.dx.item()),
                "dy_hr_px": float(geometry.dy.item()),
                "rotation_deg": float(geometry.theta_deg.item()),
                "local_max_hr_px": float(torch.linalg.vector_norm(geometry.local_field, dim=1).amax().item()),
                "min_jacobian": float(jacobian_determinant(geometry.local_field).amin().item()),
                "metrics": metrics,
            }
            per_case.append(record)
            if args.print_cases:
                print(f"CASE={case_idx+1:02d} dx={record['dx_hr_px']:+.3f} dy={record['dy_hr_px']:+.3f} "
                      f"rot={record['rotation_deg']:+.3f} local={record['local_max_hr_px']:.3f} | {format_metrics(metrics)}")

    warp_metrics = average_metrics(warp_rows)

    print("=" * 118)
    print("UAFL_FINAL_HSI_DEFORMED_TEST")
    for k, v in test_conditions.items():
        print(f"  {k}={v}")
    print("-" * 118)
    print(f"REGISTERED {format_metrics(registered_metrics)}")
    print(f"WARP       {format_metrics(warp_metrics)}")
    print("-" * 118)
    print("TABLE")
    print("Group       PSNR       SSIM      ERGAS        SAM         CC    RMSE(x255)")
    print(f"Registered  {registered_metrics['PSNR']:8.4f}  {registered_metrics['SSIM']:9.6f}  "
          f"{registered_metrics['ERGAS']:9.4f}  {registered_metrics['SAM']:9.4f}  "
          f"{registered_metrics['CC']:9.6f}  {registered_metrics['RMSE_x255']:10.4f}")
    print(f"Warp        {warp_metrics['PSNR']:8.4f}  {warp_metrics['SSIM']:9.6f}  "
          f"{warp_metrics['ERGAS']:9.4f}  {warp_metrics['SAM']:9.4f}  "
          f"{warp_metrics['CC']:9.6f}  {warp_metrics['RMSE_x255']:10.4f}")
    print("=" * 118)

    output = {
        "test_conditions": test_conditions,
        "average_metrics": {
            "Registered": registered_metrics,
            "Warp": warp_metrics,
        },
        "per_case": per_case,
    }
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"FINAL_TEST_JSON={out}")


if __name__ == "__main__":
    main()
