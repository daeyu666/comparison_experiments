"""SA-RFM network wrapper: preserve the released ImprovedDiT/metric/ODE design.

Source: https://github.com/ZhangQuan-hub/SA-RFM/blob/main/SB/sb.py
Upstream network and losses are mirrored in upstream_sb.py.
This adapter only adds a stable HSI/MSI signature and memory-safe Euler inference.
The released training objective is L_RFM + lambda_otd * L_OTD; gradient and SAM
losses are monitored upstream but NOT included in its actual total_loss.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

if __package__:
    from .upstream_sb import (
        ImprovedDiT,
        riemannian_flow_matching_loss,
        optimal_transport_direction_loss,
    )
else:
    from upstream_sb import (
        ImprovedDiT,
        riemannian_flow_matching_loss,
        optimal_transport_direction_loss,
    )

ARCHITECTURE_ID = "SA-RFM-official-ImprovedDiT-v1"
UPSTREAM_BLOB_SHA = "bb1f9c472d9539c73c4f3bd3d296990b8e16b2de"


class SARFMModel(nn.Module):
    """Official structure-aware velocity field + Euler ODE fusion."""

    def __init__(
        self, hsi_channels: int, msi_channels: int, *,
        dim: int = 220, alpha: float = 1.0, beta: float = 10.0,
        sigma: float = 1.0, ode_steps: int = 1,
    ):
        super().__init__()
        if dim < 4 or dim % max(1, dim // 20):
            raise ValueError("Official grouped-fusion dimension requires dim % max(1, dim//20) == 0")
        if ode_steps < 1:
            raise ValueError("ode_steps must be >= 1")
        if alpha <= 0 or beta < 0 or sigma <= 0:
            raise ValueError("Expected alpha>0, beta>=0 and sigma>0")
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.sigma = float(sigma)
        self.ode_steps = int(ode_steps)
        self.net = ImprovedDiT(
            hsi_channels=self.hsi_channels,
            msi_channels=self.msi_channels,
            dim=self.dim,
            alpha=self.alpha,
            beta=self.beta,
            sigma=self.sigma,
        )

    def velocity(self, x_t: torch.Tensor, t: torch.Tensor,
                 lr_hsi: torch.Tensor, hr_msi: torch.Tensor):
        return self.net(x_t, t, lr_hsi, hr_msi)

    def flow_loss(self, lr_hsi: torch.Tensor, gt: torch.Tensor,
                  hr_msi: torch.Tensor, *, lambda_otd: float = 1.0, valid_mask=None):
        """Actual effective objective from upstream SA_SB_Trainer.train_step."""
        if lambda_otd < 0:
            raise ValueError("lambda_otd must be nonnegative")
        x0 = F.interpolate(lr_hsi, size=gt.shape[-2:], mode="bicubic", align_corners=False)
        b = int(gt.shape[0])
        t = torch.rand(b, device=gt.device, dtype=gt.dtype)
        x_t = (1.0 - t[:, None, None, None]) * x0 + t[:, None, None, None] * gt
        u_target = gt - x0
        result = self.velocity(x_t, t, lr_hsi, hr_msi)
        v, metric = result["pred"], result["metric_tensor"]
        valid = None if valid_mask is None else valid_mask > 0.5
        loss_rfm = riemannian_flow_matching_loss(v, u_target, metric, use_riemannian=True, valid_mask=valid)
        loss_otd = optimal_transport_direction_loss(v, u_target, metric, valid_mask=valid)
        total = loss_rfm + lambda_otd * loss_otd
        return total, {"rfm": loss_rfm.detach(), "otd": loss_otd.detach()}

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor) -> torch.Tensor:
        """Released ode_solve Euler update, without storing each step on CPU."""
        if lr_hsi.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("Inputs must have B,C,H,W shape")
        if lr_hsi.shape[0] != hr_msi.shape[0]:
            raise ValueError("HSI/MSI batch mismatch")
        if lr_hsi.shape[1] != self.hsi_channels or hr_msi.shape[1] != self.msi_channels:
            raise ValueError("HSI/MSI channel mismatch")
        x = F.interpolate(lr_hsi, size=hr_msi.shape[-2:], mode="bicubic", align_corners=False)
        for k in range(self.ode_steps):
            t = torch.full(
                (lr_hsi.shape[0],), k / self.ode_steps,
                dtype=x.dtype, device=x.device
            )
            v = self.velocity(x, t, lr_hsi, hr_msi)["pred"]
            x = x + v / self.ode_steps
        return x


def build_model(info, args, device):
    return SARFMModel(
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        dim=int(getattr(args, "dim", 220)),
        alpha=float(getattr(args, "alpha", 1.0)),
        beta=float(getattr(args, "beta", 10.0)),
        sigma=float(getattr(args, "sigma", 1.0)),
        ode_steps=int(getattr(args, "ode_steps", 1)),
    ).to(device)
