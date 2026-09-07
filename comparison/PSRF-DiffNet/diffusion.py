"""Diffusion wrapper for the PSRF-DiffNet reproduction.

The released implementation trains a network to predict clean HR-HSI x0 from
noisy HR-HSI, LR-HSI and HR-MSI. This module keeps that objective while fixing
hard-coded dimensions/devices and the released reverse-process state update.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from network import CCFNet, CoarseRegistrationNetwork, build_registration_patches


@dataclass
class PSRFForwardOutput:
    loss: torch.Tensor
    prediction: torch.Tensor
    coarse_theta: torch.Tensor
    fine_offset_px: torch.Tensor
    timestep: torch.Tensor


class PSRFDiffusion(nn.Module):
    """PSRF registration/fusion network under an x0-prediction diffusion loss."""

    def __init__(
        self,
        *,
        patch_size: int,
        scale_ratio: int,
        hsi_channels: int,
        msi_channels: int,
        n_timestep: int = 2000,
        linear_start: float = 1e-4,
        linear_end: float = 2e-3,
    ):
        super().__init__()
        if patch_size % scale_ratio:
            raise ValueError("patch_size must be divisible by scale_ratio")
        if n_timestep < 2:
            raise ValueError("n_timestep must be >= 2")

        self.patch_size = int(patch_size)
        self.scale_ratio = int(scale_ratio)
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.n_timestep = int(n_timestep)

        self.crn = CoarseRegistrationNetwork(
            hr_patch_size=patch_size,
            scale_ratio=scale_ratio,
            hsi_channels=hsi_channels,
            msi_channels=msi_channels,
        )
        self.ccf = CCFNet(
            patch_size=patch_size,
            hsi_channels=hsi_channels,
            msi_channels=msi_channels,
        )

        betas = torch.linspace(
            float(linear_start), float(linear_end), self.n_timestep,
            dtype=torch.float64,
        )
        alphas = 1.0 - betas
        alpha_bar = torch.cumprod(alphas, dim=0)
        alpha_bar_prev = torch.cat([torch.ones(1, dtype=torch.float64), alpha_bar[:-1]])

        posterior_variance = betas * (1.0 - alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef1 = betas * torch.sqrt(alpha_bar_prev) / (1.0 - alpha_bar)
        posterior_mean_coef2 = (
            (1.0 - alpha_bar_prev) * torch.sqrt(alphas) / (1.0 - alpha_bar)
        )

        for name, value in (
            ("betas", betas),
            ("alphas", alphas),
            ("alpha_bar", alpha_bar),
            ("alpha_bar_prev", alpha_bar_prev),
            ("posterior_variance", posterior_variance),
            ("posterior_mean_coef1", posterior_mean_coef1),
            ("posterior_mean_coef2", posterior_mean_coef2),
        ):
            self.register_buffer(name, value.float())

    @staticmethod
    def _extract(values: torch.Tensor, t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        out = values.gather(0, t)
        return out.view(t.shape[0], *((1,) * (x.ndim - 1)))

    def _prepare_conditions(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if lr_hsi.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("lr_hsi and hr_msi must be BxCxHxW")
        if lr_hsi.shape[1] != self.hsi_channels:
            raise ValueError(
                f"expected {self.hsi_channels} HSI channels, got {lr_hsi.shape[1]}"
            )
        if hr_msi.shape[1] != self.msi_channels:
            raise ValueError(
                f"expected {self.msi_channels} MSI channels, got {hr_msi.shape[1]}"
            )
        if hr_msi.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"expected HR patch {self.patch_size}x{self.patch_size}, "
                f"got {tuple(hr_msi.shape[-2:])}"
            )

        lr_up = F.interpolate(
            lr_hsi,
            size=(self.patch_size, self.patch_size),
            mode="bicubic",
            align_corners=False,
        )
        lr_msi = F.interpolate(
            hr_msi,
            size=lr_hsi.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )
        lr_reg, theta = self.crn(lr_up, lr_hsi, lr_msi)
        hsi_patch, msi_patch2 = build_registration_patches(lr_up, hr_msi)
        return lr_reg, theta, hsi_patch, msi_patch2

    def predict_x0(
        self,
        x_t: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        *,
        prepared: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if prepared is None:
            prepared = self._prepare_conditions(lr_hsi, hr_msi)
        lr_reg, theta, hsi_patch, msi_patch2 = prepared
        pred, fine_offset = self.ccf(
            torch.cat([lr_reg, x_t], dim=1),
            hr_msi,
            hsi_patch,
            msi_patch2,
        )
        return pred, theta, fine_offset

    def training_loss(
        self,
        gt: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> PSRFForwardOutput:
        if gt.shape[-2:] != (self.patch_size, self.patch_size):
            raise ValueError(
                f"training GT must be {self.patch_size}x{self.patch_size}; "
                f"got {tuple(gt.shape[-2:])}"
            )
        b = gt.shape[0]
        t = torch.randint(
            1,
            self.n_timestep,
            (b,),
            generator=generator,
            device="cpu",
        ).to(gt.device)
        noise = torch.randn(
            gt.shape,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(gt.device, dtype=gt.dtype)
        alpha = self._extract(self.alpha_bar, t, gt)
        x_t = alpha.sqrt() * gt + (1.0 - alpha).sqrt() * noise

        prepared = self._prepare_conditions(lr_hsi, hr_msi)
        pred, theta, fine_offset = self.predict_x0(
            x_t, lr_hsi, hr_msi, prepared=prepared
        )
        loss = F.l1_loss(pred, gt)
        return PSRFForwardOutput(loss, pred, theta, fine_offset, t)

    def forward(
        self,
        gt: torch.Tensor,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        *,
        generator: Optional[torch.Generator] = None,
    ) -> PSRFForwardOutput:
        return self.training_loss(gt, lr_hsi, hr_msi, generator=generator)

    @torch.no_grad()
    def sample_ddim(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        *,
        sample_steps: int = 200,
        generator: Optional[torch.Generator] = None,
        clamp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Correct deterministic DDIM sampling for an x0-prediction model."""
        if sample_steps < 1:
            raise ValueError("sample_steps must be >= 1")
        sample_steps = min(int(sample_steps), self.n_timestep)
        b = lr_hsi.shape[0]
        x = torch.rand(
            b,
            self.hsi_channels,
            self.patch_size,
            self.patch_size,
            generator=generator,
            device="cpu",
            dtype=torch.float32,
        ).to(lr_hsi.device, dtype=lr_hsi.dtype)

        prepared = self._prepare_conditions(lr_hsi, hr_msi)
        times = torch.linspace(
            self.n_timestep - 1,
            0,
            sample_steps,
            device=lr_hsi.device,
        ).round().long().unique(sorted=True).flip(0)

        last_theta = None
        last_offset = None
        for i, t_scalar in enumerate(times):
            t_value = int(t_scalar.item())
            pred_x0, last_theta, last_offset = self.predict_x0(
                x, lr_hsi, hr_msi, prepared=prepared
            )
            if clamp:
                pred_x0 = pred_x0.clamp(0.0, 1.0)

            if t_value == 0 or i == len(times) - 1:
                x = pred_x0
                continue

            next_t = int(times[i + 1].item())
            a_t = self.alpha_bar[t_value].to(x.dtype)
            a_next = self.alpha_bar[next_t].to(x.dtype)
            eps = (x - a_t.sqrt() * pred_x0) / (1.0 - a_t).sqrt().clamp_min(1e-8)
            x = a_next.sqrt() * pred_x0 + (1.0 - a_next).sqrt() * eps

        diagnostics = {
            "coarse_theta": last_theta,
            "fine_offset_px": last_offset,
        }
        return x, diagnostics

    @torch.no_grad()
    def sample_tiled(
        self,
        lr_hsi: torch.Tensor,
        hr_msi: torch.Tensor,
        *,
        sample_steps: int = 200,
        tile_stride: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        clamp: bool = True,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Run fixed-patch PSRF over a larger validation/test image by overlap-add."""
        if hr_msi.ndim != 4 or lr_hsi.ndim != 4:
            raise ValueError("lr_hsi and hr_msi must be BxCxHxW")
        if hr_msi.shape[0] != 1 or lr_hsi.shape[0] != 1:
            raise ValueError("sample_tiled currently expects batch size 1")
        h, w = hr_msi.shape[-2:]
        if (h, w) == (self.patch_size, self.patch_size):
            return self.sample_ddim(
                lr_hsi, hr_msi, sample_steps=sample_steps,
                generator=generator, clamp=clamp
            )
        stride = self.patch_size if tile_stride is None else int(tile_stride)
        if stride < 1 or stride > self.patch_size:
            raise ValueError("tile_stride must be in [1, patch_size]")
        if stride % self.scale_ratio or self.patch_size % self.scale_ratio:
            raise ValueError("tile stride/size must be divisible by scale_ratio")

        def positions(length: int):
            if length < self.patch_size:
                raise ValueError("image is smaller than PSRF training patch")
            out = list(range(0, length - self.patch_size + 1, stride))
            last = length - self.patch_size
            if out[-1] != last:
                out.append(last)
            if any(v % self.scale_ratio for v in out):
                raise ValueError("tile origins must align to LR grid")
            return out

        ys, xs = positions(h), positions(w)
        output = torch.zeros(
            1, self.hsi_channels, h, w, device=hr_msi.device, dtype=hr_msi.dtype
        )
        weight = torch.zeros(1, 1, h, w, device=hr_msi.device, dtype=hr_msi.dtype)
        fine_means = []
        coarse_thetas = []
        lr_tile = self.patch_size // self.scale_ratio

        for top in ys:
            for left in xs:
                lt, ll = top // self.scale_ratio, left // self.scale_ratio
                msi_tile = hr_msi[..., top:top+self.patch_size, left:left+self.patch_size]
                hsi_tile = lr_hsi[..., lt:lt+lr_tile, ll:ll+lr_tile]
                pred, diag = self.sample_ddim(
                    hsi_tile, msi_tile, sample_steps=sample_steps,
                    generator=generator, clamp=clamp
                )
                output[..., top:top+self.patch_size, left:left+self.patch_size] += pred
                weight[..., top:top+self.patch_size, left:left+self.patch_size] += 1.0
                coarse_thetas.append(diag["coarse_theta"])
                fine_means.append(
                    torch.linalg.vector_norm(diag["fine_offset_px"], dim=-1).mean()
                )

        output = output / weight.clamp_min(1.0)
        diagnostics = {
            "coarse_theta": torch.cat(coarse_thetas, dim=0),
            "fine_offset_mean_px": torch.stack(fine_means).mean(),
        }
        return output, diagnostics


__all__ = ["PSRFDiffusion", "PSRFForwardOutput"]
