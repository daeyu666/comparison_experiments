# Shared synthetic benchmark helpers; copied from the existing PSRF-DiffNet adapter.
# Stage geometry, SRF and data configuration remain identical to comparison methods.

"""Shared experiment helpers for the PMI-RFCoNet comparison."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import TrainConfig, get_dataset_configs  # noqa: E402
from experiment_protocol import DATASETS  # noqa: E402
from comparison.UAFL.common import require_srf_weights  # noqa: E402
from comparison.UAFL.hsi_deformation import (  # noqa: E402
    make_deformed_lr_hsi, sample_synthetic_geometry, sample_training_geometry_batch,
)


def add_data_args(parser) -> None:
    parser.add_argument("--dataset", choices=DATASETS, default="PaviaU")
    parser.add_argument("--data_root", default="./data/raw")
    for name, default in (("image_size", 128), ("patch_size", 64), ("stride", 32), ("scale_ratio", 4)):
        parser.add_argument("--" + name, type=int, default=default)
    parser.add_argument("--degradation_mode", choices=["physical", "gaussian_bicubic"], default="physical")
    parser.add_argument("--degradation_sigma", type=float, default=2.0)
    parser.add_argument("--degradation_kernel_size", type=int, default=5)
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)
    parser.add_argument("--msi_mode", choices=["srf"], default="srf")
    parser.add_argument("--srf_path", default="")
    parser.add_argument("--wavelength_root", default="./data/wavelengths")
    parser.add_argument("--wavelength_path", default="")
    parser.add_argument("--srf_interp", choices=["pchip", "linear"], default="pchip")
    parser.add_argument("--srf_band_set", default="auto")
    parser.add_argument("--device", default="cuda:0")


def add_geometry_args(parser) -> None:
    parser.add_argument("--max_translation", type=float, default=4.0)
    parser.add_argument("--max_rotation_deg", type=float, default=2.0)
    parser.add_argument("--max_local_px", type=float, default=4.0)
    parser.add_argument("--control_grid", type=int, default=5)
    parser.add_argument("--min_jacobian", type=float, default=0.5)
    parser.add_argument("--registered_probability", type=float, default=0.10)


def make_generator(device, seed: int) -> torch.Generator:
    return torch.Generator(device=device).manual_seed(int(seed))


def geometry_kwargs(args) -> Dict[str, float]:
    return {
        "max_translation": float(args.max_translation),
        "max_rotation_deg": float(args.max_rotation_deg),
        "max_local_px": float(args.max_local_px),
        "control_grid": int(args.control_grid),
        "min_jacobian": float(args.min_jacobian),
    }


def deformed_observation(gt: torch.Tensor, p0, args, generator: torch.Generator, *, mixed: bool):
    """Formal shared protocol: warp HR-HSI before P0; leave HR-MSI untouched."""
    if not mixed:
        return p0.degrade(gt), int(gt.shape[0])
    identity = torch.rand(gt.shape[0], device=gt.device, generator=generator) < float(args.registered_probability)
    if bool(identity.all()):
        return p0.degrade(gt), int(identity.sum().item())
    geometry = sample_training_geometry_batch(
        gt.shape[0], *gt.shape[-2:], device=gt.device, dtype=gt.dtype,
        generator=generator, **geometry_kwargs(args),
    )
    warped = make_deformed_lr_hsi(gt, geometry, p0)
    if bool(identity.any()):
        warped = torch.where(identity[:, None, None, None], p0.degrade(gt), warped)
    return warped, int(identity.sum().item())


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(requested: str) -> torch.device:
    if str(requested).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu for a CPU check")
    return torch.device(requested)


def build_shared_cfg(args) -> TrainConfig:
    """Build the root comparison_experiments data configuration."""
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()
    for name in (
        "dataset",
        "data_root",
        "image_size",
        "patch_size",
        "stride",
        "scale_ratio",
        "degradation_mode",
        "degradation_sigma",
        "degradation_kernel_size",
        "mtf_nyquist",
        "psf_truncate",
        "msi_mode",
        "srf_path",
        "wavelength_root",
        "wavelength_path",
        "srf_interp",
        "srf_band_set",
        "batch_size",
        "num_workers",
        "seed",
        "device",
    ):
        if hasattr(args, name):
            setattr(cfg, name, getattr(args, name))
    dataset_cfg = cfg.datasets[cfg.dataset]
    cfg.n_select_bands = dataset_cfg.n_select_bands
    cfg.validation_size = int(cfg.image_size)
    return cfg


