"""Shared data, geometry, checkpoint and RNG helpers for MoE-PNP."""
from __future__ import annotations

import hashlib
import importlib.util
import random
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison.UAFL.common import build_shared_cfg, require_srf_weights, set_seed
from comparison.UAFL.hsi_deformation import (
    jacobian_determinant,
    make_deformed_lr_hsi,
    sample_synthetic_geometry,
    sample_training_geometry_batch,
)
from experiment_protocol import DATASETS, DATASET_VALIDATION_INTERVAL

_MODEL_SPEC = importlib.util.spec_from_file_location("moe_pnp_model", HERE / "model.py")
if _MODEL_SPEC is None or _MODEL_SPEC.loader is None:
    raise ImportError("Cannot load MoE-PNP model.py")
_MODEL_MODULE = importlib.util.module_from_spec(_MODEL_SPEC)
sys.modules[_MODEL_SPEC.name] = _MODEL_MODULE
_MODEL_SPEC.loader.exec_module(_MODEL_MODULE)

MoEPNP = _MODEL_MODULE.MoEPNP
UPSTREAM_COMMIT = _MODEL_MODULE.UPSTREAM_COMMIT
architecture_metadata = _MODEL_MODULE.architecture_metadata
parameter_count = _MODEL_MODULE.parameter_count

PROTOCOL_KEYS = (
    "dataset", "scale_ratio", "degradation_mode", "mtf_nyquist", "psf_truncate",
    "degradation_sigma", "degradation_kernel_size", "msi_mode", "srf_path",
    "wavelength_path", "wavelength_root", "srf_interp", "srf_band_set",
    "image_size", "patch_size", "stride",
)
GEOMETRY_KEYS = (
    "max_translation", "max_rotation_deg", "max_local_px",
    "control_grid", "min_jacobian", "registered_probability",
)


def add_data_args(parser):
    parser.add_argument("--dataset", choices=DATASETS, default="PaviaU")
    parser.add_argument("--data_root", default="./data/raw")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
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


def add_geometry_args(parser):
    parser.add_argument("--max_translation", type=float, default=4.0)
    parser.add_argument("--max_rotation_deg", type=float, default=2.0)
    parser.add_argument("--max_local_px", type=float, default=4.0)
    parser.add_argument("--control_grid", type=int, default=5)
    parser.add_argument("--min_jacobian", type=float, default=0.5)
    parser.add_argument("--registered_probability", type=float, default=0.10)


def resolve_device(name):
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu explicitly for a CPU check")
    return torch.device(name)


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def model_for_info(info, args, device):
    srf = require_srf_weights(info)
    model = MoEPNP(
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        scale_ratio=int(args.scale_ratio),
        image_size=int(args.image_size),
    ).to(device)
    model.sensor_fingerprint = hashlib.sha256(
        torch.as_tensor(srf, dtype=torch.float32).cpu().numpy().tobytes()
    ).hexdigest()
    return model


def geometry_kwargs(args):
    return {
        name: getattr(args, name)
        for name in GEOMETRY_KEYS
        if name != "registered_probability" and hasattr(args, name)
    }


def deformed_observation(gt, degradation, args, generator, *, mixed):
    """Deform HR-HSI before P0; keep HR-MSI fixed in the reliable reference frame."""
    if not mixed:
        return degradation.degrade(gt), int(gt.shape[0])
    identity = torch.rand(
        gt.shape[0], device=gt.device, generator=generator
    ) < float(args.registered_probability)
    if bool(identity.all()):
        return degradation.degrade(gt), int(identity.sum().item())
    geometry = sample_training_geometry_batch(
        gt.shape[0], *gt.shape[-2:],
        device=gt.device, dtype=gt.dtype, generator=generator,
        **geometry_kwargs(args),
    )
    warped = make_deformed_lr_hsi(gt, geometry, degradation)
    if bool(identity.any()):
        warped = torch.where(
            identity[:, None, None, None], degradation.degrade(gt), warped
        )
    return warped, int(identity.sum().item())


def snapshot_rng(generator=None):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "geometry": generator.get_state() if generator is not None else None,
    }


def restore_rng(state, generator=None):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])
    if generator is not None and state.get("geometry") is not None:
        generator.set_state(state["geometry"].cpu())


def checkpoint_payload(model, optimizer, epoch, best_psnr, bad_validations, args, generator):
    return {
        "method": "MoE-PNP",
        **architecture_metadata(model),
        "sensor_fingerprint": getattr(model, "sensor_fingerprint", None),
        "dataset": args.dataset,
        "training_stage": args.stage,
        "registered_probability": 1.0 if args.stage == "registered" else float(args.registered_probability),
        "args": vars(args).copy(),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_psnr": float(best_psnr),
        "bad_validations": int(bad_validations),
        "rng": snapshot_rng(generator),
    }


def verify_checkpoint(checkpoint, model, args, *, expected_stage):
    if checkpoint.get("method") != "MoE-PNP":
        raise ValueError("MoE-PNP checkpoint method mismatch")
    for key in ("architecture_id", "upstream_commit", "hsi_channels", "msi_channels", "scale_ratio"):
        if checkpoint.get(key) != getattr(model, key):
            raise ValueError(f"MoE-PNP checkpoint architecture mismatch: {key}")
    if checkpoint.get("dataset") != args.dataset:
        raise ValueError("MoE-PNP checkpoint/test dataset mismatch")
    if checkpoint.get("training_stage") != expected_stage:
        raise ValueError(f"MoE-PNP checkpoint stage mismatch: expected {expected_stage}")
    fingerprint = getattr(model, "sensor_fingerprint", None)
    if fingerprint is not None and checkpoint.get("sensor_fingerprint") != fingerprint:
        raise ValueError("MoE-PNP checkpoint sensor SRF mismatch")

    recorded = checkpoint.get("args", {})
    keys = PROTOCOL_KEYS + (GEOMETRY_KEYS if expected_stage == "mixed" else ())
    for key in keys:
        if hasattr(args, key) and key in recorded and recorded.get(key) != getattr(args, key):
            raise ValueError(
                f"MoE-PNP checkpoint protocol mismatch for {key}: "
                f"{recorded.get(key)!r} != {getattr(args, key)!r}"
            )


def load_checkpoint(path, model, args, *, expected_stage, optimizer=None):
    checkpoint = torch.load(
        path, map_location=next(model.parameters()).device, weights_only=False
    )
    verify_checkpoint(checkpoint, model, args, expected_stage=expected_stage)
    model.load_state_dict(checkpoint["model"], strict=True)
    if optimizer is not None:
        if checkpoint.get("optimizer") is None or "rng" not in checkpoint:
            raise ValueError("Exact resume requires optimizer and RNG state")
        recorded = checkpoint.get("args", {})
        for key in ("lr", "weight_decay", "batch_size", "seed", "validation_cases", "validation_interval"):
            if hasattr(args, key) and recorded.get(key) != getattr(args, key):
                raise ValueError(f"Resume schedule mismatch for {key}")
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
