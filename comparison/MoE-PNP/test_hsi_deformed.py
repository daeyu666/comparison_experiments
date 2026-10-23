"""MoE-PNP registered/mixed evaluation adapter for the shared x4 benchmark."""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_COMMON_SPEC = importlib.util.spec_from_file_location("moe_pnp_common", THIS_DIR / "common.py")
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise ImportError("Cannot load MoE-PNP common.py")
if "moe_pnp_common" in sys.modules:
    common = sys.modules["moe_pnp_common"]
else:
    common = importlib.util.module_from_spec(_COMMON_SPEC)
    sys.modules[_COMMON_SPEC.name] = common
    _COMMON_SPEC.loader.exec_module(common)

from data_loader import build_datasets
from metrics import MetricAverager, calc_metrics


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    common.add_data_args(p)
    common.add_geometry_args(p)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--checkpoint", default="")
    p.add_argument("--test_mode", choices=["registered_only", "registered_and_warp"], default="registered_and_warp")
    p.add_argument("--print_cases", action="store_true")
    p.add_argument("--output_json", default="")
    return p


def build_test_components(args, expected_stage=None):
    stage = expected_stage or (
        "registered" if getattr(args, "test_mode", "registered_and_warp") == "registered_only"
        else getattr(args, "mode", "mixed")
    )
    args.stage = stage
    device = common.resolve_device(args.device)
    _, _, test_set, info = build_datasets(common.build_shared_cfg(args), include_validation=True)
    if len(test_set) < 1:
        raise ValueError("Empty MoE-PNP test split")
    degradation = test_set.degradation_operator.to(device)
    model = common.model_for_info(info, args, device)
    checkpoint_path = Path(args.checkpoint) if getattr(args, "checkpoint", "") else (
        THIS_DIR / "checkpoints" / ("physical" if stage == "registered" else "hsi_warp_final")
        / args.dataset / "best.pth.tar"
    )
    checkpoint_path = checkpoint_path.resolve()
    checkpoint = common.load_checkpoint(checkpoint_path, model, args, expected_stage=stage)
    args.checkpoint = str(checkpoint_path)
    model.eval()
    return test_set, model, degradation, device, checkpoint


def main():
    args = parser().parse_args()
    if args.test_mode == "registered_and_warp" and args.cases < 1:
        raise ValueError("--cases must be >=1")
    if (args.image_size, args.patch_size, args.stride, args.scale_ratio) != (128, 64, 32, 4):
        raise ValueError("Unified test protocol is fixed at image_size=128, patch_size=64, stride=32, x4")

    common.set_seed(args.seed)
    stage = "registered" if args.test_mode == "registered_only" else "mixed"
    args.stage = stage
    test_set, model, degradation, device, checkpoint = build_test_components(args, expected_stage=stage)
    print(f"MOE_PNP_TEST stage={stage} epoch={checkpoint['epoch']} checkpoint={args.checkpoint}", flush=True)

    registered_meter, warp_meter = MetricAverager(), MetricAverager()
    per_case = []
    cuda_devices = []
    if device.type == "cuda":
        cuda_devices = [device.index if device.index is not None else torch.cuda.current_device()]

    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.eval_seed)

        for sample_idx in range(len(test_set)):
            item = test_set[sample_idx]
            gt = item["gt"].unsqueeze(0).to(device, dtype=torch.float32)
            hr_msi = item["hr_msi"].unsqueeze(0).to(device, dtype=torch.float32)
            registered_lr = degradation.degrade(gt)
            registered_pred = model(registered_lr, hr_msi)
            registered_meter.update(calc_metrics(registered_pred, gt, args.scale_ratio))

            if args.test_mode == "registered_and_warp":
                geometry_generator = common.make_generator(device, args.seed + 70000)
                for case_idx in range(args.cases):
                    geometry = common.sample_synthetic_geometry(
                        gt.shape[-2], gt.shape[-1],
                        device=device, dtype=gt.dtype, generator=geometry_generator,
                        max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg,
                        max_local_px=args.max_local_px,
                        control_grid=args.control_grid,
                        min_jacobian=args.min_jacobian,
                        local_strength_min_fraction=0.0,
                        local_strength_max_fraction=1.0,
                    )
                    warped_lr = common.make_deformed_lr_hsi(gt, geometry, degradation)
                    prediction = model(warped_lr, hr_msi)
                    metrics = calc_metrics(prediction, gt, args.scale_ratio)
                    warp_meter.update(metrics)
                    record = {
                        "sample": sample_idx + 1,
                        "case": case_idx + 1,
                        "dx_hr_px": float(geometry.dx.item()),
                        "dy_hr_px": float(geometry.dy.item()),
                        "rotation_deg": float(geometry.theta_deg.item()),
                        "local_max_hr_px": float(torch.linalg.vector_norm(geometry.local_field, dim=1).amax().item()),
                        "min_jacobian": float(common.jacobian_determinant(geometry.local_field).amin().item()),
                        "metrics": metrics,
                    }
                    per_case.append(record)
                    if args.print_cases:
                        print(
                            f"SAMPLE={sample_idx + 1:04d} CASE={case_idx + 1:02d} "
                            f"dx={record['dx_hr_px']:+.3f} dy={record['dy_hr_px']:+.3f} "
                            f"rot={record['rotation_deg']:+.3f} local={record['local_max_hr_px']:.3f} "
                            f"PSNR={metrics['PSNR']:.4f} SAM={metrics['SAM']:.4f}",
                            flush=True,
                        )

    result = {
        "method": "MoE-PNP",
        "dataset": args.dataset,
        "mode": stage,
        "checkpoint": args.checkpoint,
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "registered_metrics": registered_meter.average(),
        "warp_metrics": warp_meter.average() if per_case else None,
        "per_case": per_case,
        "protocol": {
            "degradation_mode": args.degradation_mode,
            "scale_ratio": args.scale_ratio,
            "image_size": args.image_size,
            "patch_size": args.patch_size,
            "stride": args.stride,
            "registered_probability": args.registered_probability,
            "geometry": {
                "max_translation": args.max_translation,
                "max_rotation_deg": args.max_rotation_deg,
                "max_local_px": args.max_local_px,
                "control_grid": args.control_grid,
                "min_jacobian": args.min_jacobian,
            },
        },
    }
    output_path = Path(args.output_json) if args.output_json else (
        THIS_DIR / "outputs" / "unified_two_stage" / args.dataset / f"MoEPNP_{stage}_seed{args.seed}.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"MOE_PNP_RESULT registered={result['registered_metrics']} "
        f"warp={result['warp_metrics']} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
