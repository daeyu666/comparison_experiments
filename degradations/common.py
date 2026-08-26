"""Shared low-level operators for comparison degradation experiments."""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F


def validate_hsi_tensor(x: torch.Tensor) -> None:
    if x.ndim != 4:
        raise ValueError(f"Expected BxCxHxW tensor, got shape={tuple(x.shape)}")
    if not torch.is_floating_point(x):
        raise TypeError(f"Expected floating tensor, got dtype={x.dtype}")


def validate_scale(x: torch.Tensor, scale: int) -> None:
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    h, w = x.shape[-2:]
    if h % scale != 0 or w % scale != 0:
        raise ValueError(
            f"Spatial size {(h, w)} must be divisible by scale={scale}"
        )


def gaussian_kernel2d(
    sigma: float,
    *,
    dtype: torch.dtype,
    device: torch.device,
    kernel_size: Optional[int] = None,
    truncate: float = 3.0,
) -> Optional[torch.Tensor]:
    if sigma <= 1e-8:
        return None

    if kernel_size is None:
        radius = max(1, int(math.ceil(truncate * sigma)))
        kernel_size = 2 * radius + 1
    else:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        radius = kernel_size // 2

    coords = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    g = torch.exp(-(coords * coords) / (2.0 * sigma * sigma))
    g = g / g.sum().clamp_min(torch.finfo(dtype).eps)
    kernel = torch.outer(g, g)
    return kernel / kernel.sum().clamp_min(torch.finfo(dtype).eps)


def depthwise_psf(
    x: torch.Tensor,
    sigma: float,
    *,
    kernel_size: Optional[int] = None,
    truncate: float = 3.0,
) -> torch.Tensor:
    validate_hsi_tensor(x)
    kernel = gaussian_kernel2d(
        sigma,
        dtype=x.dtype,
        device=x.device,
        kernel_size=kernel_size,
        truncate=truncate,
    )
    if kernel is None:
        return x

    c = x.shape[1]
    k = kernel.shape[-1]
    weight = kernel.view(1, 1, k, k).repeat(c, 1, 1, 1)
    return F.conv2d(x, weight, padding=k // 2, groups=c)


def area_average_downsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Detector pixel-area integration followed by stride sampling."""
    validate_hsi_tensor(x)
    validate_scale(x, scale)
    if scale == 1:
        return x
    return F.avg_pool2d(x, kernel_size=scale, stride=scale)


def resize_down(
    x: torch.Tensor,
    scale: int,
    *,
    mode: str = "bicubic",
    antialias: bool = True,
) -> torch.Tensor:
    validate_hsi_tensor(x)
    validate_scale(x, scale)
    if scale == 1:
        return x
    h, w = x.shape[-2:]
    kwargs = {}
    if mode in ("bilinear", "bicubic"):
        kwargs["align_corners"] = False
        kwargs["antialias"] = antialias
    return F.interpolate(
        x, size=(h // scale, w // scale), mode=mode, **kwargs
    )
