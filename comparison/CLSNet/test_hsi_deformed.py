"""CLSNet final test: same held-out patches, cases and formulas as EMR-Diff."""
from __future__ import annotations
import argparse
import json
import random
import sys
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
THIS_DIR = Path(__file__).resolve().parent
from comparison.CLSNet.common import (
    HERE, add_data_args, add_geometry_args, build_shared_cfg, load_checkpoint,
    make_generator, model_for_info, resolve_device, make_deformed_lr_hsi,
    sample_synthetic_geometry,
)
from comparison.UAFL.hsi_deformation import jacobian_determinant
from comparison.CLSNet.final_metrics import calc_metrics, average_metrics, format_metrics, summarize_geometry_cases
from comparison.CLSNet.model import architecture_metadata
from data_loader import build_datasets


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument('--seed', type=int, default=10)
    p.add_argument('--cases', type=int, default=10)
    p.add_argument('--eval_seed', type=int, default=1234)
    p.add_argument('--checkpoint', default='')
    p.add_argument('--test_mode', choices=['registered_only','registered_and_warp'], default='registered_and_warp')
    p.add_argument('--print_cases', action='store_true')
    p.add_argument('--output_json', default='')
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.test_mode == "registered_and_warp" and args.cases < 1:
        raise ValueError("--cases must be >=1")
    if (
        args.image_size != 128 or args.patch_size != 64 or args.stride != 32
        or args.scale_ratio != 4
    ):
        raise ValueError("Unified test protocol is fixed at 128/64/32/x4")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = resolve_device(args.device)
    _, _, test_set, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(test_set):
        raise ValueError('Empty CLSNet test split')
    p0 = test_set.degradation_operator.to(device)
    model = model_for_info(info, args, device)
    stage = 'registered' if args.test_mode == 'registered_only' else 'mixed'
    family = args.degradation_mode if stage == 'registered' else 'hsi_warp_final'
    ckpt_path = Path(args.checkpoint) if args.checkpoint else HERE/'checkpoints'/family/args.dataset/'best.pth.tar'
    checkpoint = load_checkpoint(ckpt_path, model, args, expected_stage=stage)
    model.eval()
    print(f'CLSNET_TEST_CHECKPOINT stage={stage} epoch={checkpoint["epoch"]} path={ckpt_path}')

    registered_rows, warp_rows, per_case = [], [], []
    cuda_devices = []
    if device.type == "cuda":
        idx = (
            device.index
            if device.index is not None
            else torch.cuda.current_device()
        )
        cuda_devices = [idx]

    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.eval_seed)

        for sample_idx in range(len(test_set)):
            item = test_set[sample_idx]
            gt = item["gt"].unsqueeze(0).to(
                device, dtype=torch.float32
            )
            hr_msi = item["hr_msi"].unsqueeze(0).to(
                device, dtype=torch.float32
            )

            registered_lr = p0.degrade(gt)
            registered_pred = model(registered_lr, hr_msi)
            registered_rows.append(
                calc_metrics(registered_pred, gt, args.scale_ratio)
            )

            if args.test_mode == "registered_and_warp":
                geometry_gen = make_generator(
                    device, args.seed + 70000
                )
                for case_idx in range(args.cases):
                    geometry = sample_synthetic_geometry(
                        gt.shape[-2],
                        gt.shape[-1],
                        device=device,
                        dtype=gt.dtype,
                        generator=geometry_gen,
                        max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg,
                        max_local_px=args.max_local_px,
                        control_grid=args.control_grid,
                        min_jacobian=args.min_jacobian,
                        local_strength_min_fraction=0.0,
                        local_strength_max_fraction=1.0,
                    )
                    warped_lr = make_deformed_lr_hsi(gt, geometry, p0)
                    pred = model(warped_lr, hr_msi)
                    metrics = calc_metrics(pred, gt, args.scale_ratio)
                    warp_rows.append(metrics)
                    record = {
                        "sample": sample_idx + 1,
                        "case": case_idx + 1,
                        "dx_hr_px": float(geometry.dx.item()),
                        "dy_hr_px": float(geometry.dy.item()),
                        "rotation_deg": float(geometry.theta_deg.item()),
                        "local_max_hr_px": float(
                            torch.linalg.vector_norm(
                                geometry.local_field, dim=1
                            ).amax().item()
                        ),
                        "min_jacobian": float(
                            jacobian_determinant(
                                geometry.local_field
                            ).amin().item()
                        ),
                        "metrics": metrics,
                    }
                    per_case.append(record)
                    if args.print_cases:
                        print(
                            f"SAMPLE={sample_idx+1:04d} CASE={case_idx+1:02d} "
                            f"dx={record['dx_hr_px']:+.3f} "
                            f"dy={record['dy_hr_px']:+.3f} "
                            f"rot={record['rotation_deg']:+.3f} "
                            f"local={record['local_max_hr_px']:.3f} | "
                            f"{format_metrics(metrics)}"
                        )

    reg = average_metrics(registered_rows)
    warp = average_metrics(warp_rows) if warp_rows else None
    geometry_summary = summarize_geometry_cases(per_case)
    conditions = {
        "method": "CLSNet",
        **architecture_metadata(model),
        "dataset": args.dataset,
        "test_mode": args.test_mode,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_training_stage": checkpoint.get("training_stage"),
        "checkpoint_registered_probability": checkpoint.get("registered_probability"),
        "checkpoint_architecture_id": checkpoint.get("architecture_id"),
        "test_patch_count": len(test_set),
        "registered_observations": len(registered_rows),
        "warp_observations": len(warp_rows),
        "metric_aggregation": "arithmetic mean over patches/cases, not pooled MSE",
        "rmse_scale": "RMSE is raw [0,1]; RMSE_x255 is display only",
        "ssim_definition": "simplified global-statistics SSIM",
        "formal_mixed_identity_probability": 0.10,
        "formal_mixed_deformed_probability": 0.90,
        "warp_dx_dy": f"independent U(-{args.max_translation:g},{args.max_translation:g}) HR pixels",
        "warp_rotation": f"U(-{args.max_rotation_deg:g},{args.max_rotation_deg:g}) degrees",
        "warp_local_amplitude": f"U(0,{args.max_local_px:g}) proposal subject to min Jacobian {args.min_jacobian:g}",
        "geometry_generator_seed": args.seed + 70000,
        "geometry_generator_device": str(device),
        "geometry_schedule": "generator reset for each patch; same cases reused",
        "warp_observation": "Y_H=P0(W_phi(X)); HR-MSI and GT remain registered",
        "scale_ratio": args.scale_ratio,
        "degradation_mode": args.degradation_mode,
        "mtf_nyquist": args.mtf_nyquist,
        "psf_truncate": args.psf_truncate,
        "max_translation_hr_px": args.max_translation,
        "max_rotation_deg": args.max_rotation_deg,
        "max_local_hr_px": args.max_local_px,
        "local_strength_min_fraction": 0.0,
        "local_strength_max_fraction": 1.0,
        "control_grid": args.control_grid,
        "min_local_jacobian": args.min_jacobian,
        "cases_per_patch": args.cases if warp is not None else 0,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
    }

    print("=" * 118)
    print("CLSNET_UNIFIED_TWO_STAGE_TEST")
    print(f"REGISTERED {format_metrics(reg)}")
    if warp is not None:
        print(f"WARP       {format_metrics(warp)}")
        local = geometry_summary["local_max_hr_px"]
        print(
            f"WARP_STRENGTH distinct_cases={geometry_summary['distinct_geometry_cases']} "
            f"local_max_hr_px min/mean/max="
            f"{local['min']:.3f}/{local['mean']:.3f}/{local['max']:.3f}; "
            f"test_patches={len(test_set)} warp_observations={len(warp_rows)}"
        )
    print("=" * 118)

    output = {
        "test_conditions": conditions,
        "geometry_summary": geometry_summary,
        "average_metrics": (
            {"Registered": reg, "Warp": warp}
            if warp is not None else {"Registered": reg}
        ),
        "per_sample_case": per_case,
    }
    if args.output_json:
        out = Path(args.output_json)
    else:
        tag = "registered" if warp is None else "mixed"
        out = (
            THIS_DIR / "outputs" / "unified_two_stage" / args.dataset
            / f"CLSNet_{tag}_seed{args.seed}.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINAL_TEST_JSON={out}")


if __name__ == "__main__":
    main()
