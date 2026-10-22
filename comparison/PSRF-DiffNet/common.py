"""Shared experiment helpers for the PSRF-DiffNet comparison."""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from config import TrainConfig, get_dataset_configs  # noqa: E402
from degradations import make_misaligned_msi  # noqa: E402
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


MISALIGNMENT_MODES = (
    "registered",
    "translation",
    "rotation",
    "global",
    "local",
    "global_local",
)


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


def resolve_misalignment_kwargs(
    mode: str,
    *,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
) -> Dict[str, float]:
    mode = str(mode).lower().strip()
    if mode not in MISALIGNMENT_MODES:
        raise ValueError(f"unsupported misalignment mode: {mode}")
    if mode == "registered":
        return dict(
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=0.0,
        )
    if mode == "translation":
        return dict(
            translation_max_px=float(translation_max_px),
            rotation_max_deg=0.0,
            local_max_displacement_px=0.0,
        )
    if mode == "rotation":
        return dict(
            translation_max_px=0.0,
            rotation_max_deg=float(rotation_max_deg),
            local_max_displacement_px=0.0,
        )
    if mode == "global":
        return dict(
            translation_max_px=float(translation_max_px),
            rotation_max_deg=float(rotation_max_deg),
            local_max_displacement_px=0.0,
        )
    if mode == "local":
        return dict(
            translation_max_px=0.0,
            rotation_max_deg=0.0,
            local_max_displacement_px=float(local_max_displacement_px),
        )
    return dict(
        translation_max_px=float(translation_max_px),
        rotation_max_deg=float(rotation_max_deg),
        local_max_displacement_px=float(local_max_displacement_px),
    )


def warp_msi_for_protocol(
    hr_msi: torch.Tensor,
    *,
    mode: str,
    translation_max_px: float,
    rotation_max_deg: float,
    local_max_displacement_px: float,
    control_grid_size: int,
    generator: torch.Generator,
):
    kwargs = resolve_misalignment_kwargs(
        mode,
        translation_max_px=translation_max_px,
        rotation_max_deg=rotation_max_deg,
        local_max_displacement_px=local_max_displacement_px,
    )
    return make_misaligned_msi(
        hr_msi,
        control_grid_size=int(control_grid_size),
        generator=generator,
        **kwargs,
    )


def calc_masked_psnr_sam(
    pred: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    *,
    threshold: float = 0.999,
    eps: float = 1e-8,
) -> Tuple[float, float, float]:
    """Same valid-overlap PSNR/SAM definition used by S2Diff diagnostics."""
    if pred.shape[0] != 1 or target.shape[0] != 1 or valid_mask.shape[0] != 1:
        raise ValueError("masked metric helper expects batch size 1")
    mask = valid_mask[0, 0] >= float(threshold)
    valid_count = int(mask.sum().item())
    if valid_count < 1:
        raise ValueError("no valid pixels remain after MSI warp")

    pred_clamped = pred.detach().float().clamp(0.0, 1.0)[0, :, mask]
    target_clamped = target.detach().float().clamp(0.0, 1.0)[0, :, mask]
    mse = torch.mean((pred_clamped - target_clamped) ** 2).item()
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / max(mse, 1e-12))

    pred_spec = pred.detach().float()[0, :, mask]
    target_spec = target.detach().float()[0, :, mask]
    dot = torch.sum(pred_spec * target_spec, dim=0)
    pred_norm = torch.sqrt(torch.sum(pred_spec.square(), dim=0) + eps)
    target_norm = torch.sqrt(torch.sum(target_spec.square(), dim=0) + eps)
    cos = (dot / (pred_norm * target_norm + eps)).clamp(-1.0 + eps, 1.0 - eps)
    sam = torch.mean(torch.acos(cos) * 180.0 / math.pi).item()
    return float(psnr), float(sam), float(valid_count / mask.numel())


def checkpoint_payload(model, optimizer, epoch: int, best_psnr: float, args) -> Dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_psnr": float(best_psnr),
        "args": vars(args),
    }


__all__ = [
    "MISALIGNMENT_MODES",
    "add_data_args",
    "add_geometry_args",
    "build_shared_cfg",
    "deformed_observation",
    "geometry_kwargs",
    "make_deformed_lr_hsi",
    "make_generator",
    "require_srf_weights",
    "sample_synthetic_geometry",
    "set_seed",
    "calc_masked_psnr_sam",
    "checkpoint_payload",
    "resolve_device",
    "resolve_misalignment_kwargs",
    "warp_msi_for_protocol",
]
