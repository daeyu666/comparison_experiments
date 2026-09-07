"""Shared benchmark helpers for the HSIFN comparison."""

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
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU")
        return torch.device("cpu")
    return torch.device(requested)


def build_shared_cfg(args) -> TrainConfig:
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
        return dict(translation_max_px=0.0, rotation_max_deg=0.0, local_max_displacement_px=0.0)
    if mode == "translation":
        return dict(translation_max_px=float(translation_max_px), rotation_max_deg=0.0, local_max_displacement_px=0.0)
    if mode == "rotation":
        return dict(translation_max_px=0.0, rotation_max_deg=float(rotation_max_deg), local_max_displacement_px=0.0)
    if mode == "global":
        return dict(translation_max_px=float(translation_max_px), rotation_max_deg=float(rotation_max_deg), local_max_displacement_px=0.0)
    if mode == "local":
        return dict(translation_max_px=0.0, rotation_max_deg=0.0, local_max_displacement_px=float(local_max_displacement_px))
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
    if pred.shape[0] != 1 or target.shape[0] != 1 or valid_mask.shape[0] != 1:
        raise ValueError("masked metric helper expects batch size 1")
    mask = valid_mask[0, 0] >= float(threshold)
    valid_count = int(mask.sum().item())
    if valid_count < 1:
        raise ValueError("no valid pixels remain after MSI warp")

    p = pred.detach().float().clamp(0.0, 1.0)[0, :, mask]
    t = target.detach().float().clamp(0.0, 1.0)[0, :, mask]
    mse = torch.mean((p - t) ** 2).item()
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / max(mse, 1e-12))

    p = pred.detach().float()[0, :, mask]
    t = target.detach().float()[0, :, mask]
    dot = torch.sum(p * t, dim=0)
    pn = torch.sqrt(torch.sum(p.square(), dim=0) + eps)
    tn = torch.sqrt(torch.sum(t.square(), dim=0) + eps)
    cos = (dot / (pn * tn + eps)).clamp(-1.0 + eps, 1.0 - eps)
    sam = torch.mean(torch.acos(cos) * 180.0 / math.pi).item()
    return float(psnr), float(sam), float(valid_count / mask.numel())


def require_srf_weights(info) -> torch.Tensor:
    weights = info.get("srf_weights")
    if weights is None:
        raise RuntimeError(
            "HSIFN needs the sensor SRF to synthesize an HSI-side MSI proxy. "
            "Run with --msi_mode srf (the shared benchmark default)."
        )
    return torch.as_tensor(weights, dtype=torch.float32)


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
    "build_shared_cfg",
    "calc_masked_psnr_sam",
    "checkpoint_payload",
    "require_srf_weights",
    "resolve_device",
    "set_seed",
    "warp_msi_for_protocol",
]
