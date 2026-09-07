"""Shared benchmark helpers for the PRFCoAM reproduction."""

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


def require_srf_weights(info) -> torch.Tensor:
    weights = info.get("srf_weights")
    if weights is None:
        raise RuntimeError(
            "PRFCoAM reproduction requires shared sensor SRF weights. "
            "Run with --msi_mode srf (default)."
        )
    return torch.as_tensor(weights, dtype=torch.float32)


def checkpoint_payload(model, optimizer, scheduler, epoch: int, best_psnr: float, args) -> Dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "epoch": int(epoch),
        "best_psnr": float(best_psnr),
        "args": vars(args),
    }


__all__ = [
    "build_shared_cfg",
    "checkpoint_payload",
    "require_srf_weights",
    "resolve_device",
    "set_seed",
]
