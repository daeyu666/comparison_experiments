"""Diagnose whether CLSNet actually uses LR-HSI spatial correspondence.

HR-MSI is held fixed. Starting from one base LR-HSI observation, evaluate:
normal, pixel_shuffle, patch_shuffle, mean_broadcast, zeros, and other_sample.

The base LR-HSI may be registered or synthesized with the repository's shared
fixed-deformation protocol. This is a modality/spatial-dependence diagnostic,
not a formal benchmark table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch

from comparison.CLSNet.common import (
    HERE,
    add_data_args,
    add_geometry_args,
    build_shared_cfg,
    load_checkpoint,
    model_for_info,
    resolve_device,
)
from comparison.CLSNet.model import architecture_metadata
from data_loader import build_datasets
from fixed_deformation_protocol import (
    METRIC_KEYS,
    STRENGTHS,
    calc_metrics,
    sample_fixed_geometry,
    warp_hsi,
)


CONDITIONS = (
    "normal",
    "pixel_shuffle",
    "patch_shuffle",
    "mean_broadcast",
    "zeros",
    "other_sample",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--mode", choices=("registered", "mixed"), default="mixed")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=1)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument(
        "--source_deformation",
        choices=("registered", "bspline", "elastic", "tarc", "tarc_physical"),
        default="tarc",
        help="Base LR-HSI observation used before dependency perturbation.",
    )
    p.add_argument("--level", type=int, choices=range(1, 7), default=5)
    p.add_argument("--elastic_sigma_hr_px", type=float, default=8.0)
    p.add_argument("--elastic_truncate", type=float, default=3.0)
    p.add_argument("--tarc_alpha", type=float, default=1000.0)
    p.add_argument("--tarc_sigma_hr_px", type=float, default=2.0)
    p.add_argument("--patch_shuffle_size", type=int, default=4)
    p.add_argument("--output_dir", default="")
    return p.parse_args(argv)


def checkpoint_path(args):
    if args.checkpoint:
        return Path(args.checkpoint)
    family = args.degradation_mode if args.mode == "registered" else "hsi_warp_final"
    return HERE / "checkpoints" / family / args.dataset / "best.pth.tar"


def cpu_generator(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))


def pixel_shuffle(lr_hsi, *, seed):
    """Destroy LR spatial correspondence while preserving each pixel spectrum."""
    b, c, h, w = lr_hsi.shape
    if b != 1:
        raise ValueError("dependency diagnostic expects batch size 1")
    perm = torch.randperm(h * w, generator=cpu_generator(seed)).to(lr_hsi.device)
    flat = lr_hsi.reshape(b, c, h * w)
    return flat.index_select(2, perm).reshape_as(lr_hsi)


def patch_shuffle(lr_hsi, *, seed, patch_size):
    """Shuffle LR blocks while preserving content inside each block."""
    b, c, h, w = lr_hsi.shape
    patch_size = int(patch_size)
    if b != 1:
        raise ValueError("dependency diagnostic expects batch size 1")
    if patch_size < 1 or h % patch_size or w % patch_size:
        raise ValueError(
            f"patch_shuffle_size={patch_size} must divide LR shape {h}x{w}"
        )
    gh, gw = h // patch_size, w // patch_size
    blocks = (
        lr_hsi.reshape(b, c, gh, patch_size, gw, patch_size)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(b, gh * gw, c, patch_size, patch_size)
    )
    perm = torch.randperm(gh * gw, generator=cpu_generator(seed)).to(lr_hsi.device)
    blocks = blocks.index_select(1, perm)
    return (
        blocks.reshape(b, gh, gw, c, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, c, h, w)
    )


def perturb(lr_hsi, condition, *, seed, patch_size, donor=None):
    if condition == "normal":
        return lr_hsi
    if condition == "pixel_shuffle":
        return pixel_shuffle(lr_hsi, seed=seed)
    if condition == "patch_shuffle":
        return patch_shuffle(lr_hsi, seed=seed, patch_size=patch_size)
    if condition == "mean_broadcast":
        return lr_hsi.mean(dim=(-2, -1), keepdim=True).expand_as(lr_hsi)
    if condition == "zeros":
        return torch.zeros_like(lr_hsi)
    if condition == "other_sample":
        if donor is None:
            raise ValueError("other_sample requires donor LR-HSI")
        return donor
    raise ValueError(condition)


def synthesize_base_lr(gt, p0, args, *, sample_index, case_index):
    if args.source_deformation == "registered":
        return p0.degrade(gt), {
            "local_deformation": "registered",
            "dx_hr_px": 0.0,
            "dy_hr_px": 0.0,
            "rotation_deg": 0.0,
        }

    strength = STRENGTHS[args.level - 1]
    geometry, record = sample_fixed_geometry(
        *gt.shape[-2:],
        strength,
        sample_index=sample_index,
        case_index=case_index,
        seed=args.seed,
        control_grid=args.control_grid,
        min_jacobian=args.min_jacobian,
        local_deformation=args.source_deformation,
        elastic_sigma_hr_px=args.elastic_sigma_hr_px,
        elastic_truncate=args.elastic_truncate,
        tarc_alpha=args.tarc_alpha,
        tarc_sigma_hr_px=args.tarc_sigma_hr_px,
    )
    return p0.degrade(warp_hsi(gt, geometry)), record


def average(rows):
    return {key: mean(row[key] for row in rows) for key in METRIC_KEYS}


def main(argv=None):
    args = parse_args(argv)
    if args.cases < 1:
        raise ValueError("--cases must be >=1")
    if (
        args.image_size != 128
        or args.patch_size != 64
        or args.stride != 32
        or args.scale_ratio != 4
    ):
        raise ValueError("Unified comparison protocol is fixed at 128/64/32/x4")

    device = resolve_device(args.device)
    _, val_set, test_set, info = build_datasets(
        build_shared_cfg(args), include_validation=True
    )
    if len(test_set) < 1:
        raise ValueError("Empty test split")
    if len(test_set) == 1 and len(val_set) < 1:
        raise ValueError("Single-patch test split needs a validation donor for other_sample")

    p0 = test_set.degradation_operator.to(device)
    model = model_for_info(info, args, device)
    ckpt = checkpoint_path(args)
    checkpoint = load_checkpoint(ckpt, model, args, expected_stage=args.mode)
    model.eval()

    tag = args.source_deformation
    if args.source_deformation.startswith("tarc"):
        tag += f"_L{args.level}_a{args.tarc_alpha:g}_s{args.tarc_sigma_hr_px:g}"
    elif args.source_deformation != "registered":
        tag += f"_L{args.level}"

    output_dir = Path(
        args.output_dir
        or HERE / "outputs" / "lrhsi_dependency" / args.dataset / f"{args.mode}_{tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    for sample_index in range(len(test_set)):
        item = test_set[sample_index]
        samples.append(
            (
                item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32),
                item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32),
            )
        )

    # PaviaU/Houston/Botswana center-128 protocols contain one test item.
    # For their other_sample diagnostic, use the held-out validation region as
    # a donor instead of leaking a training patch. Multi-item test sets use the
    # next test item as the donor.
    validation_donor = None
    if len(samples) == 1:
        item = val_set[0]
        validation_donor = item["gt"].unsqueeze(0).to(
            device=device, dtype=torch.float32
        )

    rows_by_condition = {name: [] for name in CONDITIONS}
    detail_rows = []

    with torch.no_grad():
        for case_index in range(args.cases):
            base_lrs = []
            geometry_records = []
            for sample_index, (gt, _hr_msi) in enumerate(samples):
                lr_hsi, geometry_record = synthesize_base_lr(
                    gt,
                    p0,
                    args,
                    sample_index=sample_index,
                    case_index=case_index,
                )
                base_lrs.append(lr_hsi)
                geometry_records.append(geometry_record)

            validation_donor_lr = None
            if validation_donor is not None:
                validation_donor_lr, _ = synthesize_base_lr(
                    validation_donor,
                    p0,
                    args,
                    sample_index=1000000,
                    case_index=case_index,
                )

            for sample_index, (gt, hr_msi) in enumerate(samples):
                base_lr = base_lrs[sample_index]
                if len(samples) > 1:
                    donor_index = (sample_index + 1) % len(samples)
                    donor_lr = base_lrs[donor_index]
                    donor_label = f"test:{donor_index + 1}"
                else:
                    donor_index = None
                    donor_lr = validation_donor_lr
                    donor_label = "validation:1"
                perturb_seed = (
                    int(args.eval_seed)
                    + sample_index * 1000003
                    + case_index * 9176
                ) % (2**63 - 1)

                for condition in CONDITIONS:
                    test_lr = perturb(
                        base_lr,
                        condition,
                        seed=perturb_seed,
                        patch_size=args.patch_shuffle_size,
                        donor=donor_lr,
                    )
                    prediction = model(test_lr, hr_msi)
                    if prediction.shape != gt.shape or not torch.isfinite(prediction).all():
                        raise ValueError("Prediction must be finite and have GT HSI shape")
                    metrics = calc_metrics(prediction, gt, args.scale_ratio)
                    rows_by_condition[condition].append(metrics)
                    detail_rows.append(
                        {
                            "sample": sample_index + 1,
                            "case": case_index + 1,
                            "condition": condition,
                            "donor_sample": donor_label
                            if condition == "other_sample"
                            else None,
                            "perturb_seed": perturb_seed,
                            "geometry": geometry_records[sample_index],
                            "metrics": metrics,
                        }
                    )

    averages = {name: average(rows_by_condition[name]) for name in CONDITIONS}
    normal = averages["normal"]
    deltas = {
        name: {
            "delta_PSNR_vs_normal": averages[name]["PSNR"] - normal["PSNR"],
            "delta_SAM_vs_normal": averages[name]["SAM"] - normal["SAM"],
        }
        for name in CONDITIONS
    }

    conditions = {
        "method": "CLSNet",
        **architecture_metadata(model),
        "dataset": args.dataset,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "checkpoint_training_stage": checkpoint.get("training_stage"),
        "checkpoint_registered_probability": checkpoint.get("registered_probability"),
        "source_deformation": args.source_deformation,
        "level": None if args.source_deformation == "registered" else args.level,
        "tarc_alpha": args.tarc_alpha
        if args.source_deformation.startswith("tarc")
        else None,
        "tarc_sigma_hr_px": args.tarc_sigma_hr_px
        if args.source_deformation.startswith("tarc")
        else None,
        "elastic_sigma_hr_px": args.elastic_sigma_hr_px
        if args.source_deformation == "elastic"
        else None,
        "patch_shuffle_size_lr_px": args.patch_shuffle_size,
        "cases_per_patch": args.cases,
        "test_patch_count": len(test_set),
        "ablation_conditions": list(CONDITIONS),
        "pixel_shuffle": (
            "shared spatial permutation across all HSI bands; per-pixel spectra preserved"
        ),
        "patch_shuffle": (
            "non-overlapping LR blocks permuted; within-block spectra/spatial structure preserved"
        ),
        "mean_broadcast": "per-band LR spatial mean broadcast to all LR positions",
        "other_sample": (
            "next test LR-HSI paired with current HR-MSI; for single-item center-128 "
            "test splits, use the held-out validation region as donor"
        ),
        "seed": args.seed,
        "eval_seed": args.eval_seed,
    }

    payload = {
        "test_conditions": conditions,
        "average_metrics": averages,
        "deltas_vs_normal": deltas,
        "per_sample_case_condition": detail_rows,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )

    csv_path = output_dir / "summary.csv"
    fields = [
        "condition",
        *METRIC_KEYS,
        "delta_PSNR_vs_normal",
        "delta_SAM_vs_normal",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for name in CONDITIONS:
            writer.writerow({"condition": name, **averages[name], **deltas[name]})

    print("=" * 104)
    print(
        f"CLSNET_LRHSI_DEPENDENCY dataset={args.dataset} mode={args.mode} "
        f"source={args.source_deformation} checkpoint={ckpt}"
    )
    for name in CONDITIONS:
        m = averages[name]
        d = deltas[name]
        print(
            f"{name:14s} "
            f"PSNR={m['PSNR']:.6f} SAM={m['SAM']:.6f} "
            f"dPSNR={d['delta_PSNR_vs_normal']:+.6f} "
            f"dSAM={d['delta_SAM_vs_normal']:+.6f}"
        )
    print("=" * 104)
    print(f"LRHSI_DEPENDENCY_SUMMARY={csv_path}")


if __name__ == "__main__":
    main()
