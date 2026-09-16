"""UAFL-side HSI acquisition deformation matched to S2Diff-MH.

Formal geometry protocol:
  X (reliable MSI coordinates)
    -> W_phi(X) on HR-HSI
    -> fixed physical spatial degradation P0
    -> deformed LR-HSI observation Y_H

The HR-MSI reference remains registered/reliable.  This file intentionally lives
inside comparison/UAFL so the shared comparison code and S2Diff-MH are untouched.

The synthetic geometry matches S2Diff-MH's final protocol:
  dx, dy ~ U(-max_translation, +max_translation) independently
  theta ~ U(-max_rotation_deg, +max_rotation_deg)
  local deformation: zero-mean KxK sparse controls -> cubic B-spline dense field
  test local strength in [0.65, 1.0] * max_local_px
  non-folding constraint: min Jacobian >= min_jacobian
  forward sampling uses border padding and align_corners=True
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F


@dataclass
class SyntheticGeometry:
    dx: torch.Tensor
    dy: torch.Tensor
    theta_deg: torch.Tensor
    control: torch.Tensor
    local_field: torch.Tensor


def _bspline_basis_matrix(
    n_ctrl: int,
    n_out: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if n_ctrl < 2 or n_out < 1:
        raise ValueError("n_ctrl must be >=2 and n_out must be >=1")
    u = torch.linspace(0.0, float(n_ctrl - 1), n_out, device=device, dtype=dtype)
    base = torch.floor(u).to(torch.long)
    t = u - base.to(dtype)
    weights = torch.stack(
        [
            (1.0 - t).pow(3) / 6.0,
            (3.0 * t.pow(3) - 6.0 * t.pow(2) + 4.0) / 6.0,
            (-3.0 * t.pow(3) + 3.0 * t.pow(2) + 3.0 * t + 1.0) / 6.0,
            t.pow(3) / 6.0,
        ],
        dim=1,
    )
    indices = torch.stack([base - 1, base, base + 1, base + 2], dim=1).clamp(0, n_ctrl - 1)
    matrix = torch.zeros((n_out, n_ctrl), device=device, dtype=dtype)
    matrix.scatter_add_(1, indices, weights)
    return matrix


def cubic_bspline_field(control: torch.Tensor, output_size: Tuple[int, int]) -> torch.Tensor:
    if control.ndim != 4 or control.shape[1] != 2:
        raise ValueError("control must have shape Bx2xKxK")
    h, w = map(int, output_size)
    ky, kx = control.shape[-2:]
    wy = _bspline_basis_matrix(ky, h, device=control.device, dtype=control.dtype)
    wx = _bspline_basis_matrix(kx, w, device=control.device, dtype=control.dtype)
    return torch.einsum("hi,bcij,wj->bchw", wy, control, wx)


def zero_mean_control(control: torch.Tensor) -> torch.Tensor:
    return control - control.mean(dim=(-2, -1), keepdim=True)


def jacobian_determinant(local_field: torch.Tensor) -> torch.Tensor:
    vx, vy = local_field[:, 0], local_field[:, 1]
    dvx_dx = vx[:, :-1, 1:] - vx[:, :-1, :-1]
    dvx_dy = vx[:, 1:, :-1] - vx[:, :-1, :-1]
    dvy_dx = vy[:, :-1, 1:] - vy[:, :-1, :-1]
    dvy_dy = vy[:, 1:, :-1] - vy[:, :-1, :-1]
    return (1.0 + dvx_dx) * (1.0 + dvy_dy) - dvx_dy * dvy_dx


def sampling_coordinates(
    height: int,
    width: int,
    dx: torch.Tensor,
    dy: torch.Tensor,
    theta_deg: torch.Tensor,
    local_field: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if local_field.ndim != 4 or local_field.shape[1] != 2:
        raise ValueError("local_field must have shape Bx2xHxW")
    batch = local_field.shape[0]
    yy, xx = torch.meshgrid(
        torch.arange(height, device=local_field.device, dtype=local_field.dtype),
        torch.arange(width, device=local_field.device, dtype=local_field.dtype),
        indexing="ij",
    )
    xx = xx.unsqueeze(0).expand(batch, -1, -1) + local_field[:, 0]
    yy = yy.unsqueeze(0).expand(batch, -1, -1) + local_field[:, 1]

    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    angle = theta_deg * (math.pi / 180.0)
    cos_a = torch.cos(angle).view(batch, 1, 1)
    sin_a = torch.sin(angle).view(batch, 1, 1)
    x0, y0 = xx - cx, yy - cy
    sample_x = cos_a * x0 - sin_a * y0 + cx + dx.view(batch, 1, 1)
    sample_y = sin_a * x0 + cos_a * y0 + cy + dy.view(batch, 1, 1)
    return sample_x, sample_y


def forward_warp(
    image: torch.Tensor,
    geometry: SyntheticGeometry,
) -> torch.Tensor:
    """Forward-synthesize HSI acquisition geometry in reliable MSI coordinates."""
    if image.ndim != 4:
        raise ValueError("image must have shape BxCxHxW")
    h, w = image.shape[-2:]
    sample_x, sample_y = sampling_coordinates(
        h,
        w,
        geometry.dx,
        geometry.dy,
        geometry.theta_deg,
        geometry.local_field,
    )
    grid_x = 2.0 * sample_x / max(w - 1, 1) - 1.0
    grid_y = 2.0 * sample_y / max(h - 1, 1) - 1.0
    grid = torch.stack([grid_x, grid_y], dim=-1)
    return F.grid_sample(
        image,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


def _uniform_scalar(
    lo: float,
    hi: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
) -> torch.Tensor:
    return torch.empty(1, device=device, dtype=dtype).uniform_(lo, hi, generator=generator)


def sample_synthetic_geometry(
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_translation: float = 4.0,
    max_rotation_deg: float = 2.0,
    max_local_px: float = 4.0,
    control_grid: int = 5,
    min_jacobian: float = 0.5,
) -> SyntheticGeometry:
    """Exact final-test style sampler used by the matched protocol."""
    dx = _uniform_scalar(-max_translation, max_translation, device=device, dtype=dtype, generator=generator)
    dy = _uniform_scalar(-max_translation, max_translation, device=device, dtype=dtype, generator=generator)
    theta = _uniform_scalar(-max_rotation_deg, max_rotation_deg, device=device, dtype=dtype, generator=generator)

    control = None
    local = None
    for _ in range(64):
        candidate = torch.randn(
            (1, 2, control_grid, control_grid),
            generator=generator,
            device=device,
            dtype=dtype,
        )
        candidate = zero_mean_control(candidate)
        dense = cubic_bspline_field(candidate, (height, width))
        dense_norm = torch.linalg.vector_norm(dense, dim=1).amax().clamp_min(1e-8)
        strength = _uniform_scalar(0.65, 1.0, device=device, dtype=dtype, generator=generator) * float(max_local_px)
        candidate = candidate * (strength / dense_norm)
        dense = cubic_bspline_field(candidate, (height, width))
        if float(jacobian_determinant(dense).amin().item()) >= float(min_jacobian):
            control, local = candidate, dense
            break
    if control is None or local is None:
        raise RuntimeError("failed to sample a non-folding local deformation")
    return SyntheticGeometry(dx=dx, dy=dy, theta_deg=theta, control=control, local_field=local)


def _cat_geometry(items: List[SyntheticGeometry]) -> SyntheticGeometry:
    return SyntheticGeometry(
        dx=torch.cat([g.dx for g in items], dim=0),
        dy=torch.cat([g.dy for g in items], dim=0),
        theta_deg=torch.cat([g.theta_deg for g in items], dim=0),
        control=torch.cat([g.control for g in items], dim=0),
        local_field=torch.cat([g.local_field for g in items], dim=0),
    )


def sample_training_geometry_batch(
    batch_size: int,
    height: int,
    width: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
    generator: torch.Generator,
    max_translation: float = 4.0,
    max_rotation_deg: float = 2.0,
    max_local_px: float = 4.0,
    control_grid: int = 5,
    min_jacobian: float = 0.5,
) -> SyntheticGeometry:
    """S2Diff-MH-style train sampler: local amplitude covers the whole 0..max range."""
    items: List[SyntheticGeometry] = []
    for _ in range(int(batch_size)):
        local_cap = float(max_local_px)
        if max_local_px > 0.0:
            if float(torch.rand((), generator=generator, device=device).item()) < 0.10:
                local_cap = 0.0
            else:
                local_cap = float(torch.rand((), generator=generator, device=device).item()) * float(max_local_px)
        items.append(
            sample_synthetic_geometry(
                height,
                width,
                device=device,
                dtype=dtype,
                generator=generator,
                max_translation=max_translation,
                max_rotation_deg=max_rotation_deg,
                max_local_px=local_cap,
                control_grid=control_grid,
                min_jacobian=min_jacobian,
            )
        )
    return _cat_geometry(items)


def make_deformed_lr_hsi(
    gt_hr_hsi: torch.Tensor,
    geometry: SyntheticGeometry,
    physical_degradation,
) -> torch.Tensor:
    """Y_H = P0(W_phi(X)); MSI is intentionally not touched."""
    warped_hr_hsi = forward_warp(gt_hr_hsi, geometry)
    return physical_degradation.degrade(warped_hr_hsi)


__all__ = [
    "SyntheticGeometry",
    "cubic_bspline_field",
    "forward_warp",
    "jacobian_determinant",
    "make_deformed_lr_hsi",
    "sample_synthetic_geometry",
    "sample_training_geometry_batch",
    "sampling_coordinates",
    "zero_mean_control",
]
