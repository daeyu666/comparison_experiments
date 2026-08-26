"""Physical HSI degradation for shared comparison experiments."""

from __future__ import annotations

import math

import torch

from .base import BaseDegradation
from .common import area_average_downsample, depthwise_psf, validate_hsi_tensor


def sigma_from_mtf_nyquist(scale_ratio: int, mtf_nyquist: float) -> float:
    """Convert MTF at LR Nyquist to Gaussian PSF sigma on the HR grid."""
    if scale_ratio < 1:
        raise ValueError("scale_ratio must be >= 1")
    if not 0.0 < mtf_nyquist < 1.0:
        raise ValueError("mtf_nyquist must lie strictly between 0 and 1")
    return (
        float(scale_ratio)
        / math.pi
        * math.sqrt(-2.0 * math.log(float(mtf_nyquist)))
    )


class PhysicalDegradation(BaseDegradation):
    """Gaussian optical PSF followed by detector-area integration and sampling."""

    mode = "physical"

    def __init__(
        self,
        scale_ratio: int = 4,
        mtf_nyquist: float = 0.2,
        truncate: float = 3.0,
    ):
        super().__init__(scale_ratio=scale_ratio)
        if truncate <= 0:
            raise ValueError("truncate must be > 0")
        self.mtf_nyquist = float(mtf_nyquist)
        self.truncate = float(truncate)
        self.terminal_sigma = sigma_from_mtf_nyquist(
            scale_ratio, mtf_nyquist
        )

    def sigma_at_strength(self, strength: float) -> float:
        strength = float(min(max(strength, 0.0), 1.0))
        return self.terminal_sigma * strength

    def degrade_at(
        self, x: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        validate_hsi_tensor(x)
        sigma_t = self.sigma_at_strength(strength)
        optical = depthwise_psf(x, sigma_t, truncate=self.truncate)
        return area_average_downsample(optical, scale)

    def extra_repr(self) -> str:
        return (
            super().extra_repr()
            + f", mtf_nyquist={self.mtf_nyquist}, "
            f"terminal_sigma={self.terminal_sigma:.6f}, "
            f"truncate={self.truncate}"
        )
