"""Unified PSRF-DiffNet test adapter for registered and mixed checkpoints."""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    add_data_args,
    add_geometry_args,
    build_shared_cfg,
    geometry_kwargs,
    make_deformed_lr_hsi,
    make_generator,
    require_srf_weights,
    resolve_device,
    sample_synthetic_geometry,
    set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from autograd_safe import make_autograd_safe  # noqa: E402
from diffusion import PSRFDiffusion  # noqa: E402

# Load this directory's trainer by exact path. A plain "from train import"
# could resolve the repository-root launcher when this file is run directly.
_train_spec = importlib.util.spec_from_file_location(
    "psrf_diffnet_train_module", THIS_DIR / "train.py"
)
if _train_spec is None or _train_spec.loader is None:
    raise ImportError("Unable to load PSRF-DiffNet train.py")
_train_module = importlib.util.module_from_spec(_train_spec)
sys.modules[_train_spec.name] = _train_module
_train_spec.loader.exec_module(_train_module)
ARCHITECTURE_ID = _train_module.ARCHITECTURE_ID
load_checkpoint = _train_module.load_checkpoint


def parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--test_mode", choices=["registered_only", "registered_and_warp"], default="registered_and_warp")
    p.add_argument("--output_json", default="")
    p.add_argument("--sample_steps", type=int, default=50)
    p.add_argument("--n_timestep", type=int, default=2000)
    p.add_argument("--linear_start", type=float, default=1e-4)
    p.add_argument("--linear_end", type=float, default=2e-3)
    return p.parse_args(argv)


def build_model(info, args, device):
    model = PSRFDiffusion(
        patch_size=args.patch_size,
        scale_ratio=args.scale_ratio,
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        n_timestep=int(getattr(args, "n_timestep", 2000)),
        linear_start=float(getattr(args, "linear_start", 1e-4)),
        linear_end=float(getattr(args, "linear_end", 2e-3)),
    ).to(device)
    model.sensor_fingerprint = hashlib.sha256(
        require_srf_weights(info).detach().cpu().numpy().tobytes()
    ).hexdigest()
    make_autograd_safe(model)
    return model


def load_psrf_checkpoint(path, model, args, *, expected_stage):
    state = load_checkpoint(path, model, args, expected_stage=expected_stage)
    if state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("PSRF-DiffNet architecture metadata mismatch")
    return state


@torch.no_grad()
def predict(model, lr_hsi, hr_msi, *, sample_steps=50, patch_size=64):
    model.eval()
    output, _ = model.sample_tiled(
        lr_hsi, hr_msi, sample_steps=sample_steps, tile_stride=patch_size
    )
    return output


def main(argv=None):
    args = parser(argv)
    if args.test_mode == "registered_and_warp" and args.cases < 1:
        raise ValueError("--cases must be >=1")
    if args.sample_steps < 1:
        raise ValueError("--sample_steps must be >=1")
    if (args.image_size, args.patch_size, args.stride, args.scale_ratio) != (128, 64, 32, 4):
        raise ValueError("Unified synthetic test protocol is fixed at image_size=128, patch_size=64, stride=32, scale_ratio=4")

    set_seed(args.seed)
    device = resolve_device(args.device)
    _, _, test_set, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if len(test_set) == 0:
        raise ValueError("PSRF-DiffNet test split is empty")
    p0 = test_set.degradation_operator.to(device)
    model = build_model(info, args, device)
    stage = "registered" if args.test_mode == "registered_only" else "mixed"
    if not args.checkpoint:
        args.checkpoint = str(THIS_DIR / "checkpoints" / ("physical" if stage == "registered" else "hsi_warp_final") / args.dataset / "best.pth.tar")
    state = load_psrf_checkpoint(args.checkpoint, model, args, expected_stage=stage)
    model.eval()
    print(f"PSRF_DIFFNET_TEST stage={stage} epoch={state['epoch']} checkpoint={args.checkpoint}", flush=True)

    registered_rows, warp_rows, per_case = [], [], []
    geometry_gen = make_generator(device, args.seed + 70000)
    with torch.no_grad():
        # Keep diffusion initialization deterministic and independent from the
        # random geometry used to synthesize the LR-HSI observation.
        torch.manual_seed(args.eval_seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.eval_seed)
        for sample_idx in range(len(test_set)):
            item = test_set[sample_idx]
            gt = item["gt"].unsqueeze(0).to(device, dtype=torch.float32)
            msi = item["hr_msi"].unsqueeze(0).to(device, dtype=torch.float32)
            lr_registered = p0.degrade(gt)
            pred = predict(model, lr_registered, msi, sample_steps=args.sample_steps, patch_size=args.patch_size)
            reg = calc_metrics(pred, gt, args.scale_ratio)
            registered_rows.append(reg)

            if stage == "mixed":
                for case_idx in range(args.cases):
                    geometry = sample_synthetic_geometry(
                        *gt.shape[-2:], device=device, dtype=gt.dtype,
                        generator=geometry_gen, local_strength_min_fraction=0.0,
                        local_strength_max_fraction=1.0, **geometry_kwargs(args),
                    )
                    lr_warped = make_deformed_lr_hsi(gt, geometry, p0)
                    pred_warped = predict(
                        model, lr_warped, msi,
                        sample_steps=args.sample_steps, patch_size=args.patch_size,
                    )
                    metrics = calc_metrics(pred_warped, gt, args.scale_ratio)
                    warp_rows.append(metrics)
                    per_case.append({
                        "sample": sample_idx + 1,
                        "case": case_idx + 1,
                        "dx_hr_px": float(geometry.dx.item()),
                        "dy_hr_px": float(geometry.dy.item()),
                        "rotation_deg": float(geometry.theta_deg.item()),
                        "local_max_hr_px": float(torch.linalg.vector_norm(geometry.local_field, dim=1).amax().item()),
                        "metrics": metrics,
                    })

    def average(rows):
        if not rows:
            return None
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    result = {
        "method": "PSRF-DiffNet",
        "architecture_id": ARCHITECTURE_ID,
        "dataset": args.dataset,
        "stage": stage,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_epoch": int(state["epoch"]),
        "sensor_fingerprint": state.get("sensor_fingerprint"),
        "sample_steps": args.sample_steps,
        "registered": average(registered_rows),
        "warp": average(warp_rows),
        "per_case": per_case,
        "arguments": vars(args),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=2), flush=True)
    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved PSRF-DiffNet test results: {out}", flush=True)
    return result


if __name__ == "__main__":
    main()
