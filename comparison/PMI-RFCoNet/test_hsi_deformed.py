"""Unified synthetic registered / deformed-HSI evaluator for PMI-RFCoNet."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not __package__:
    __package__ = "comparison.PMI-RFCoNet"

from .common import (  # noqa: E402
    add_data_args, add_geometry_args, build_shared_cfg, geometry_kwargs,
    make_deformed_lr_hsi, make_generator, resolve_device, sample_synthetic_geometry,
    set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from .model import build_model, ARCHITECTURE_ID  # noqa: E402

from . import train as _TRAIN


def parser(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--test_mode", choices=["registered_only", "registered_and_warp"],
                   default="registered_and_warp")
    p.add_argument("--output_json", default="")
    p.add_argument("--max_samples", type=int, default=0, help="Timing/smoke only; 0 evaluates all test patches")
    return p.parse_args(argv)


def build_test_components(args, expected_stage="mixed"):
    """Also used by test_fixed_deformation.py and test_multiseed_deformation.py."""
    set_seed(int(getattr(args, "eval_seed", 1234)))
    device = resolve_device(args.device)
    _, _, test_set, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(test_set):
        raise ValueError("PMI-RFCoNet test split empty")
    family = "physical" if expected_stage == "registered" else "hsi_warp_final"
    ckpt_path = getattr(args, "checkpoint", "") or str(
        THIS_DIR / "checkpoints" / family / args.dataset / "best.pth.tar"
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model_args = state.get("model_args")
    if not isinstance(model_args, dict):
        raise ValueError("PMI-RFCoNet checkpoint is missing model hyperparameter metadata")
    model = build_model(info, SimpleNamespace(**model_args), device)
    state = _TRAIN.load_checkpoint(
        ckpt_path, model, args, expected_stage=expected_stage,
        fingerprint=_TRAIN.sensor_fingerprint(info)
    )
    model.eval()
    p0 = test_set.degradation_operator.to(device)
    return test_set, model, p0, device, state


def predict(model, lr_hsi, hr_msi):
    return model(lr_hsi, hr_msi)


@torch.no_grad()
def main(argv=None):
    args = parser(argv)
    if (args.image_size, args.patch_size, args.stride, args.scale_ratio) != (128, 64, 32, 4):
        raise ValueError("Shared PMI-RFCoNet synthetic protocol is fixed at (128,64,32,4)")
    if args.test_mode == "registered_and_warp" and args.cases < 1:
        raise ValueError("--cases must be >= 1")
    stage = "registered" if args.test_mode == "registered_only" else "mixed"
    test_set, model, p0, device, state = build_test_components(args, expected_stage=stage)
    geometry_gen = make_generator(device, args.seed + 70000)
    reg_rows, warp_rows, per_case = [], [], []
    if args.max_samples < 0:
        raise ValueError("max_samples must be nonnegative")
    count = min(args.max_samples, len(test_set)) if args.max_samples else len(test_set)
    for idx in range(count):
        item = test_set[idx]
        gt = item["gt"].unsqueeze(0).to(device, dtype=torch.float32)
        msi = item["hr_msi"].unsqueeze(0).to(device, dtype=torch.float32)
        lr = p0.degrade(gt)
        reg_rows.append(calc_metrics(predict(model, lr, msi), gt, args.scale_ratio))
        if stage == "mixed":
            for case_idx in range(args.cases):
                geometry = sample_synthetic_geometry(
                    *gt.shape[-2:], device=device, dtype=gt.dtype,
                    generator=geometry_gen,
                    local_strength_min_fraction=0.0, local_strength_max_fraction=1.0,
                    **geometry_kwargs(args),
                )
                lr_warped = make_deformed_lr_hsi(gt, geometry, p0)
                scores = calc_metrics(predict(model, lr_warped, msi), gt, args.scale_ratio)
                warp_rows.append(scores)
                per_case.append({
                    "sample": idx + 1, "case": case_idx + 1,
                    "dx_hr_px": float(geometry.dx.item()),
                    "dy_hr_px": float(geometry.dy.item()),
                    "rotation_deg": float(geometry.theta_deg.item()),
                    "local_max_hr_px": float(torch.linalg.vector_norm(
                        geometry.local_field, dim=1).amax().item()),
                    "metrics": scores,
                })

    def average(rows):
        return ({key: float(np.mean([row[key] for row in rows])) for key in rows[0]}
                if rows else None)

    result = {
        "method": "PMI-RFCoNet",
        "architecture_id": ARCHITECTURE_ID,
        "dataset": args.dataset,
        "stage": stage,
        "checkpoint": str(Path(args.checkpoint or
            THIS_DIR / "checkpoints" / ("physical" if stage == "registered" else "hsi_warp_final")
            / args.dataset / "best.pth.tar").resolve()),
        "checkpoint_epoch": int(state["epoch"]),
        "sensor_fingerprint": state.get("sensor_fingerprint"),
        "model_args": model.model_args,
        "test_samples": count, "smoke_test": bool(args.max_samples),
        "registered": average(reg_rows),
        "warp": average(warp_rows),
        "per_case": per_case,
        "arguments": vars(args),
    }
    print(json.dumps({k: v for k, v in result.items() if k != "per_case"}, indent=2), flush=True)
    for label, values in (("REGISTERED", result["registered"]), ("WARP", result["warp"])):
        if values:
            print(f"FINAL_{label} " + " ".join(f"{key}={value:.6f}" for key,value in values.items()), flush=True)
    path = Path(args.output_json) if args.output_json else THIS_DIR/"outputs"/"unified_two_stage"/args.dataset/(stage+".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(f"FINAL_TEST_JSON={path}", flush=True)
    return result


if __name__ == "__main__":
    main()
