"""Shared switchable HSI degradation and HSI-MSI misalignment operators."""

from __future__ import annotations

from typing import Any

from .base import BaseDegradation
from .gaussian_bicubic import GaussianBicubicDegradation
from .misalignment import (
    MisalignmentParameters,
    apply_misalignment,
    build_global_grid,
    build_local_grid,
    generate_smooth_local_displacement,
    make_misaligned_msi,
    sample_misalignment_parameters,
)
from .physical import PhysicalDegradation, sigma_from_mtf_nyquist


def build_degradation(
    mode: str,
    *,
    scale_ratio: int = 4,
    **kwargs: Any,
) -> BaseDegradation:
    mode = mode.lower().strip()
    if mode == "gaussian_bicubic":
        return GaussianBicubicDegradation(
            scale_ratio=scale_ratio,
            sigma=float(kwargs.pop("sigma", 2.0)),
            kernel_size=int(kwargs.pop("kernel_size", 5)),
        )
    if mode == "physical":
        return PhysicalDegradation(
            scale_ratio=scale_ratio,
            mtf_nyquist=float(kwargs.pop("mtf_nyquist", 0.2)),
            truncate=float(kwargs.pop("truncate", 3.0)),
        )
    raise ValueError(
        f"Unsupported degradation mode {mode!r}; expected "
        "gaussian_bicubic or physical"
    )


__all__ = [
    "BaseDegradation",
    "GaussianBicubicDegradation",
    "PhysicalDegradation",
    "MisalignmentParameters",
    "apply_misalignment",
    "build_degradation",
    "build_global_grid",
    "build_local_grid",
    "generate_smooth_local_displacement",
    "make_misaligned_msi",
    "sample_misalignment_parameters",
    "sigma_from_mtf_nyquist",
]
