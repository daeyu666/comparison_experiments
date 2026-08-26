# metrics.py
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


def _prepare_metric_pair(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use one consistent [0, 1] evaluation domain for every metric."""
    pred = torch.clamp(pred.detach().float(), 0.0, 1.0)
    target = torch.clamp(target.detach().float(), 0.0, 1.0)
    return pred, target


def calc_rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    pred, target = _prepare_metric_pair(pred, target)
    mse = F.mse_loss(pred, target).item()
    return math.sqrt(max(mse, 0.0))


def calc_psnr(pred: torch.Tensor, target: torch.Tensor, max_value: float = 1.0) -> float:
    rmse = calc_rmse(pred, target)
    if rmse <= 1e-12:
        return 100.0
    return 20.0 * math.log10(max_value / rmse)


def calc_sam(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-12) -> float:
    """Mean spectral angle in degrees over pixels with valid non-zero spectra."""
    pred, target = _prepare_metric_pair(pred, target)

    dot = torch.sum(pred * target, dim=1)
    pred_norm = torch.linalg.vector_norm(pred, dim=1)
    target_norm = torch.linalg.vector_norm(target, dim=1)

    valid = (pred_norm > eps) & (target_norm > eps)
    if not torch.any(valid):
        return 0.0

    denom = (pred_norm[valid] * target_norm[valid]).clamp_min(eps)
    cos = (dot[valid] / denom).clamp(-1.0, 1.0)
    angle = torch.acos(cos) * 180.0 / math.pi
    return torch.mean(angle).item()


def calc_cc(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    pred, target = _prepare_metric_pair(pred, target)

    b, c, h, w = pred.shape
    pred = pred.view(b, c, -1)
    target = target.view(b, c, -1)

    pred_mean = pred.mean(dim=2, keepdim=True)
    target_mean = target.mean(dim=2, keepdim=True)

    pred_centered = pred - pred_mean
    target_centered = target - target_mean

    numerator = torch.sum(pred_centered * target_centered, dim=2)
    denominator = torch.sqrt(
        torch.sum(pred_centered ** 2, dim=2)
        * torch.sum(target_centered ** 2, dim=2)
    ).clamp_min(eps)

    cc = numerator / denominator
    return torch.mean(cc).item()


def calc_ergas(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_ratio: int,
    eps: float = 1e-8,
) -> float:
    pred, target = _prepare_metric_pair(pred, target)

    rmse_per_band = torch.sqrt(torch.mean((pred - target) ** 2, dim=(0, 2, 3)))
    mean_target = torch.mean(target, dim=(0, 2, 3)).abs().clamp_min(eps)

    ergas = 100.0 / scale_ratio * torch.sqrt(
        torch.mean((rmse_per_band / mean_target) ** 2)
    )
    return ergas.item()


def calc_ssim_simple(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8) -> float:
    """Simplified global-statistics SSIM in the shared [0, 1] domain."""
    pred, target = _prepare_metric_pair(pred, target)

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = pred.mean()
    mu_y = target.mean()
    sigma_x = pred.var(unbiased=False)
    sigma_y = target.var(unbiased=False)
    sigma_xy = ((pred - mu_x) * (target - mu_y)).mean()

    ssim = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
        (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2) + eps
    )
    return ssim.item()


def calc_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
    scale_ratio: int,
) -> Dict[str, float]:
    return {
        "PSNR": calc_psnr(pred, target),
        "RMSE": calc_rmse(pred, target),
        "SAM": calc_sam(pred, target),
        "ERGAS": calc_ergas(pred, target, scale_ratio),
        "SSIM": calc_ssim_simple(pred, target),
        "CC": calc_cc(pred, target),
    }


class MetricAverager:
    def __init__(self):
        self.data = {}

    def update(self, metric_dict: Dict[str, float]):
        for key, value in metric_dict.items():
            if key not in self.data:
                self.data[key] = []
            self.data[key].append(float(value))

    def average(self) -> Dict[str, float]:
        return {
            key: float(np.mean(values))
            for key, values in self.data.items()
        }

    def reset(self):
        self.data = {}
