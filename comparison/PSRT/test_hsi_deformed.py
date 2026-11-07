"""PSRT registered and mixed final-test adapter for the shared x4 protocol."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in (str(ROOT), str(HERE)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from common import (  # noqa: E402
    add_data_args, add_geometry_args, build_shared_cfg, geometry_kwargs,
    make_deformed_lr_hsi, make_generator, resolve_device,
    sample_synthetic_geometry, set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from model import build_model, ARCHITECTURE_ID  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("psrt_stage_trainer", HERE / "train.py")
_TRAIN = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _TRAIN
_SPEC.loader.exec_module(_TRAIN)


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
    return p.parse_args(argv)


def build_test_components(args, expected_stage="mixed"):
    """Return shared test dataset, predictor, degradation, device and checkpoint."""
    set_seed(int(getattr(args, "eval_seed", 1234)))
    device = resolve_device(args.device)
    _, _, test_set, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(test_set):
        raise ValueError("PSRT test split is empty")
    family = "physical" if expected_stage == "registered" else "hsi_warp_final"
    ckpt_path = getattr(args, "checkpoint", "") or str(
        HERE / "checkpoints" / family / args.dataset / "best.pth.tar"
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    config = state.get("model_args")
    if not isinstance(config, dict) or "patch_size" not in config:
        raise ValueError("PSRT checkpoint has no patch-size metadata")
    model = build_model(info, SimpleNamespace(patch_size=config["patch_size"]), device)
    state = _TRAIN.load_checkpoint(
        ckpt_path, model, args, expected_stage=expected_stage,
        fingerprint=_TRAIN.sensor_fingerprint(info),
    )
    model.eval()
    p0 = test_set.degradation_operator.to(device)
    return test_set, model, p0, device, state


@torch.no_grad()
def main(argv=None):
    args = parser(argv)
    if (args.image_size, args.patch_size, args.stride, args.scale_ratio) != (128, 64, 32, 4):
        raise ValueError("The benchmark PSRT synthetic protocol is fixed at (128,64,32,4)")
    if args.test_mode == "registered_and_warp" and args.cases < 1:
        raise ValueError("PSRT mixed test cases must be positive")

    stage = "registered" if args.test_mode == "registered_only" else "mixed"
    test_set, model, p0, device, state = build_test_components(args, expected_stage=stage)
    geometry_gen = make_generator(device, args.seed + 70000)
    registered_rows, warp_rows, cases = [], [], []
    for sample_index in range(len(test_set)):
        sample = test_set[sample_index]
        gt = sample["gt"].unsqueeze(0).to(device, dtype=torch.float32)
        msi = sample["hr_msi"].unsqueeze(0).to(device, dtype=torch.float32)
        lr_registered = p0.degrade(gt)
        registered_rows.append(calc_metrics(model(lr_registered, msi), gt, args.scale_ratio))
        if stage == "mixed":
            for case_index in range(args.cases):
                geometry = sample_synthetic_geometry(
                    *gt.shape[-2:], device=device, dtype=gt.dtype,
                    generator=geometry_gen,
                    local_strength_min_fraction=0.0, local_strength_max_fraction=1.0,
                    **geometry_kwargs(args),
                )
                lr_deformed = make_deformed_lr_hsi(gt, geometry, p0)
                metric_row = calc_metrics(model(lr_deformed, msi), gt, args.scale_ratio)
                warp_rows.append(metric_row)
                cases.append({
                    "sample": sample_index + 1, "case": case_index + 1,
                    "dx_hr_px": float(geometry.dx.item()),
                    "dy_hr_px": float(geometry.dy.item()),
                    "rotation_deg": float(geometry.theta_deg.item()),
                    "local_max_hr_px": float(torch.linalg.vector_norm(
                        geometry.local_field, dim=1).amax().item()),
                    "metrics": metric_row,
                })

    def average(rows):
        if not rows:
            return None
        return {key: float(np.mean([row[key] for row in rows])) for key in rows[0]}

    result = {
        "method": "PSRT", "architecture_id": ARCHITECTURE_ID,
        "dataset": args.dataset, "stage": stage,
        "checkpoint": str(Path(args.checkpoint or (
            HERE / "checkpoints" / ("physical" if stage == "registered" else "hsi_warp_final")
            / args.dataset / "best.pth.tar"
        )).resolve()),
        "checkpoint_epoch": int(state["epoch"]),
        "upstream_source_sha": state.get("upstream_source_sha"),
        "sensor_fingerprint": state.get("sensor_fingerprint"),
        "registered": average(registered_rows),
        "warp": average(warp_rows),
        "per_case": cases,
        "arguments": vars(args),
    }
    print(json.dumps({key: value for key, value in result.items() if key != "per_case"},
                     indent=2, allow_nan=False), flush=True)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Saved PSRT evaluation to {path}", flush=True)
    return result


if __name__ == "__main__":
    main()
