"""Exact final-report formulas used by EMR-Diff's unified final test.

Kept separate from root metrics.py, which is used during training validation.
Source: EMR-Diff/test_hsi_deformed.py at comparison_experiments 8b81d658.
"""
import math
from statistics import mean
import torch
import torch.nn.functional as F

def calc_rmse(pred, target):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    return math.sqrt(max(mse, 1e-12))


def calc_psnr(pred, target):
    rmse = calc_rmse(pred, target)
    return 100.0 if rmse <= 1e-12 else 20.0 * math.log10(1.0 / rmse)


def calc_sam(pred, target, eps=1e-12):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    dot = torch.sum(pred * target, dim=1)
    pn = torch.linalg.vector_norm(pred, dim=1)
    tn = torch.linalg.vector_norm(target, dim=1)
    valid = (pn > eps) & (tn > eps)
    if not torch.any(valid):
        return 0.0
    cos = (dot[valid] / (pn[valid] * tn[valid]).clamp_min(eps)).clamp(-1, 1)
    return (torch.acos(cos) * 180.0 / math.pi).mean().item()


def calc_cc(pred, target, eps=1e-8):
    pred = pred.detach().float().view(pred.shape[0], pred.shape[1], -1)
    target = target.detach().float().view(target.shape[0], target.shape[1], -1)
    pc = pred - pred.mean(dim=2, keepdim=True)
    tc = target - target.mean(dim=2, keepdim=True)
    numerator = torch.sum(pc * tc, dim=2)
    denominator = torch.sqrt(
        torch.sum(pc ** 2, dim=2) * torch.sum(tc ** 2, dim=2) + eps
    )
    return torch.mean(numerator / (denominator + eps)).item()


def calc_ergas(pred, target, scale_ratio, eps=1e-8):
    pred = pred.detach().float()
    target = target.detach().float()
    rmse_band = torch.sqrt(torch.mean((pred-target)**2, dim=(0,2,3)) + eps)
    mean_target = torch.mean(target, dim=(0,2,3))
    return (
        100.0 / scale_ratio
        * torch.sqrt(torch.mean((rmse_band/(mean_target+eps))**2))
    ).item()


def calc_ssim_simple(pred, target, eps=1e-8):
    pred = pred.detach().float()
    target = target.detach().float()
    c1, c2 = 0.01**2, 0.03**2
    mx, my = pred.mean(), target.mean()
    vx, vy = pred.var(unbiased=False), target.var(unbiased=False)
    cxy = ((pred-mx)*(target-my)).mean()
    return (
        ((2*mx*my+c1)*(2*cxy+c2))
        / ((mx**2+my**2+c1)*(vx+vy+c2)+eps)
    ).item()


def calc_metrics(pred, target, scale_ratio):
    rmse = calc_rmse(pred, target)
    return {
        "PSNR": calc_psnr(pred, target),
        "SSIM": calc_ssim_simple(pred, target),
        "ERGAS": calc_ergas(pred, target, scale_ratio),
        "SAM": calc_sam(pred, target),
        "CC": calc_cc(pred, target),
        "RMSE": rmse,
        "RMSE_x255": rmse * 255.0,
    }


def average_metrics(rows):
    return {key: float(mean(float(row[key]) for row in rows)) for key in rows[0]}


def summarize_geometry_cases(records):
    """Summarize the fixed case schedule, counting each geometry only once.

    The geometry generator is reset for every patch. Hundreds of patch/case
    results therefore still represent only ``cases`` distinct geometries.
    """
    first_patch = [row for row in records if row["sample"] == 1]
    if not first_patch:
        return {}
    fields = (
        "dx_hr_px", "dy_hr_px", "rotation_deg", "local_max_hr_px",
        "min_jacobian",
    )
    summary = {"distinct_geometry_cases": len(first_patch)}
    for field in fields:
        values = [float(row[field]) for row in first_patch]
        summary[field] = {
            "min": min(values), "mean": mean(values), "max": max(values),
        }
    return summary


def format_metrics(m):
    return (
        f"PSNR={m['PSNR']:.4f} SSIM={m['SSIM']:.6f} "
        f"ERGAS={m['ERGAS']:.4f} SAM={m['SAM']:.4f} "
        f"CC={m['CC']:.6f} RMSE={m['RMSE_x255']:.4f} "
        f"(raw={m['RMSE']:.6f})"
    )


