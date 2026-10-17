"""LR-HSI spatial-dependence diagnostic for comparison models.

HR-MSI is held fixed. Starting from one base LR-HSI observation, evaluate:
normal, pixel_shuffle, patch_shuffle, mean_broadcast, zeros, and other_sample.

The base LR-HSI may be registered or synthesized by the repository's shared
fixed-deformation protocol. For stochastic models such as EMR-Diff, every
condition for the same sample/case uses the same reconstruction RNG seed so
metric differences isolate the LR-HSI perturbation rather than diffusion noise.

This is a diagnostic experiment, not a formal main-comparison table.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

import torch

from experiment_protocol import DATASETS, MODES, ROOT, checkpoint_path, resolve_model
from fixed_deformation_protocol import (
    METRIC_KEYS,
    STRENGTHS,
    calc_metrics,
    sample_fixed_geometry,
    warp_hsi,
)
from test_fixed_deformation import MODEL_BUILDERS


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
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", choices=MODES, default="mixed")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=1)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument(
        "--source_deformation",
        choices=("registered", "bspline", "elastic", "tarc", "tarc_physical"),
        default="tarc",
        help="Base LR-HSI observation before dependency perturbation.",
    )
    p.add_argument("--level", type=int, choices=range(1, 7), default=5)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--elastic_sigma_hr_px", type=float, default=8.0)
    p.add_argument("--elastic_truncate", type=float, default=3.0)
    p.add_argument("--tarc_alpha", type=float, default=1000.0)
    p.add_argument("--tarc_sigma_hr_px", type=float, default=2.0)
    p.add_argument(
        "--patch_shuffle_size",
        type=int,
        default=4,
        help="Square block size in LR-HSI pixels for patch_shuffle.",
    )
    p.add_argument(
        "--max_samples",
        type=int,
        default=0,
        help="0 uses the full held-out split; positive values use a deterministic stratified subset when scene metadata is available.",
    )
    p.add_argument("--output_dir", default="")
    return p.parse_args(argv)


def _cpu_generator(seed):
    return torch.Generator(device="cpu").manual_seed(int(seed) % (2**63 - 1))


def _pixel_shuffle(lr_hsi, *, seed):
    """Destroy spatial correspondence while preserving each LR pixel spectrum."""
    b, c, h, w = lr_hsi.shape
    if b != 1:
        raise ValueError("dependency diagnostic expects batch size 1")
    perm = torch.randperm(h * w, generator=_cpu_generator(seed)).to(lr_hsi.device)
    flat = lr_hsi.reshape(b, c, h * w)
    return flat.index_select(2, perm).reshape_as(lr_hsi)


def _patch_shuffle(lr_hsi, *, seed, patch_size):
    """Shuffle non-overlapping LR blocks while preserving within-block content."""
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
    perm = torch.randperm(gh * gw, generator=_cpu_generator(seed)).to(lr_hsi.device)
    blocks = blocks.index_select(1, perm)
    return (
        blocks.reshape(b, gh, gw, c, patch_size, patch_size)
        .permute(0, 3, 1, 4, 2, 5)
        .reshape(b, c, h, w)
    )


def _perturb(lr_hsi, condition, *, seed, patch_size, donor=None):
    if condition == "normal":
        return lr_hsi
    if condition == "pixel_shuffle":
        return _pixel_shuffle(lr_hsi, seed=seed)
    if condition == "patch_shuffle":
        return _patch_shuffle(lr_hsi, seed=seed, patch_size=patch_size)
    if condition == "mean_broadcast":
        return lr_hsi.mean(dim=(-2, -1), keepdim=True).expand_as(lr_hsi)
    if condition == "zeros":
        return torch.zeros_like(lr_hsi)
    if condition == "other_sample":
        if donor is None:
            raise ValueError("other_sample requires another held-out LR-HSI")
        return donor
    raise ValueError(condition)


def _synthesize_base_lr(gt, degrade, args, *, sample_index, case_index):
    if args.source_deformation == "registered":
        return degrade(gt), {
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
    return degrade(warp_hsi(gt, geometry)), record


def _average(rows):
    return {key: mean(row[key] for row in rows) for key in METRIC_KEYS}


def _select_sample_indices(test_set, max_samples):
    """Deterministic full-split or scene-stratified diagnostic subset."""
    total = len(test_set)
    max_samples = int(max_samples)
    if max_samples <= 0 or max_samples >= total:
        return list(range(total))

    samples_meta = getattr(test_set, "samples", None)
    if samples_meta and len(samples_meta) == total:
        groups = {}
        for idx, meta in enumerate(samples_meta):
            scene = meta[0] if isinstance(meta, (tuple, list)) and meta else str(idx)
            groups.setdefault(str(scene), []).append(idx)
        if len(groups) > 1:
            selected = []
            depth = 0
            scene_names = sorted(groups)
            while len(selected) < max_samples:
                added = False
                for scene in scene_names:
                    rows = groups[scene]
                    if depth < len(rows):
                        selected.append(rows[depth])
                        added = True
                        if len(selected) >= max_samples:
                            break
                if not added:
                    break
                depth += 1
            return selected

    step = max(1, total // max_samples)
    return list(range(0, total, step))[:max_samples]


def _donor_index(test_set, selected_indices, local_position):
    """Prefer a different held-out scene for CAVE-like scene datasets."""
    current_index = selected_indices[local_position]
    samples_meta = getattr(test_set, "samples", None)
    if samples_meta and len(samples_meta) == len(test_set):
        current_meta = samples_meta[current_index]
        current_scene = (
            current_meta[0]
            if isinstance(current_meta, (tuple, list)) and current_meta
            else None
        )
        if current_scene is not None:
            for offset in range(1, len(selected_indices)):
                candidate_pos = (local_position + offset) % len(selected_indices)
                candidate_index = selected_indices[candidate_pos]
                candidate_meta = samples_meta[candidate_index]
                candidate_scene = (
                    candidate_meta[0]
                    if isinstance(candidate_meta, (tuple, list)) and candidate_meta
                    else None
                )
                if candidate_scene != current_scene:
                    return candidate_pos

    return (local_position + 1) % len(selected_indices)


def _predict_with_fixed_rng(predict, lr_hsi, hr_msi, *, device, seed):
    """Use identical reconstruction randomness across all LR perturbations."""
    device = torch.device(device)
    cuda_devices = []
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cuda_devices = [idx]
    with torch.random.fork_rng(devices=cuda_devices):
        torch.random.default_generator.manual_seed(int(seed))
        if cuda_devices:
            with torch.cuda.device(device):
                torch.cuda.manual_seed(int(seed))
        return predict(lr_hsi, hr_msi)


def main(argv=None):
    args = parse_args(argv)
    if args.cases < 1:
        raise ValueError("--cases must be >=1")

    adapter = resolve_model(args.model)
    if adapter.canonical not in MODEL_BUILDERS:
        raise ValueError(
            f"{adapter.canonical} has no shared model builder in test_fixed_deformation.py"
        )

    args.checkpoint = str(
        Path(args.checkpoint).resolve()
        if args.checkpoint
        else checkpoint_path(adapter, args.dataset, args.mode)
    )
    args.data_root = str(Path(args.data_root).resolve())

    # Fixed shared benchmark/sensor protocol expected by all current builders.
    args.image_size, args.patch_size, args.stride, args.scale_ratio = 128, 64, 32, 4
    args.degradation_mode, args.mtf_nyquist, args.psf_truncate = "physical", 0.2, 3.0

    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for CPU testing")

    torch.manual_seed(args.seed)
    test_set, predict, degrade, device, details = MODEL_BUILDERS[adapter.canonical](args)
    selected_indices = _select_sample_indices(test_set, args.max_samples)
    if len(selected_indices) < 2:
        raise ValueError(
            "Dependency test needs at least two selected held-out items for other_sample."
        )

    tag = args.source_deformation
    if args.source_deformation.startswith("tarc"):
        tag += f"_L{args.level}_a{args.tarc_alpha:g}_s{args.tarc_sigma_hr_px:g}"
    elif args.source_deformation != "registered":
        tag += f"_L{args.level}"

    output_dir = Path(
        args.output_dir
        or ROOT
        / "comparison"
        / adapter.folder
        / "outputs"
        / "lrhsi_dependency"
        / args.dataset
        / f"{args.mode}_{tag}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cache only held-out inputs. CAVE has multiple test patches/scenes, so
    # other_sample is another held-out test item and never a training sample.
    samples = []
    for original_index in selected_indices:
        item = test_set[original_index]
        samples.append(
            (
                original_index,
                item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32),
                item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32),
            )
        )

    rows_by_condition = {name: [] for name in CONDITIONS}
    detail_rows = []

    with torch.no_grad():
        for case_index in range(args.cases):
            base_lrs = []
            geometry_records = []
            for sample_position, (original_index, gt, _hr_msi) in enumerate(samples):
                lr_hsi, geometry_record = _synthesize_base_lr(
                    gt,
                    degrade,
                    args,
                    sample_index=original_index,
                    case_index=case_index,
                )
                base_lrs.append(lr_hsi)
                geometry_records.append(geometry_record)

            for sample_position, (original_index, gt, hr_msi) in enumerate(samples):
                base_lr = base_lrs[sample_position]
                donor_position = _donor_index(
                    test_set, selected_indices, sample_position
                )
                donor_original_index = selected_indices[donor_position]
                donor_lr = base_lrs[donor_position]
                perturb_seed = (
                    int(args.eval_seed)
                    + original_index * 1000003
                    + case_index * 9176
                ) % (2**63 - 1)
                reconstruction_seed = (
                    int(args.eval_seed)
                    + 900000
                    + original_index * 1000003
                    + case_index * 9176
                ) % (2**63 - 1)

                for condition in CONDITIONS:
                    test_lr = _perturb(
                        base_lr,
                        condition,
                        seed=perturb_seed,
                        patch_size=args.patch_shuffle_size,
                        donor=donor_lr,
                    )
                    prediction = _predict_with_fixed_rng(
                        predict,
                        test_lr,
                        hr_msi,
                        device=device,
                        seed=reconstruction_seed,
                    )
                    if prediction.shape != gt.shape or not torch.isfinite(prediction).all():
                        raise ValueError("Prediction must be finite and have GT HSI shape")
                    metrics = calc_metrics(prediction, gt, args.scale_ratio)
                    rows_by_condition[condition].append(metrics)
                    detail_rows.append(
                        {
                            "sample": original_index + 1,
                            "selected_position": sample_position + 1,
                            "case": case_index + 1,
                            "condition": condition,
                            "donor_sample": donor_original_index + 1
                            if condition == "other_sample"
                            else None,
                            "perturb_seed": perturb_seed,
                            "reconstruction_seed": reconstruction_seed,
                            "geometry": geometry_records[sample_index],
                            "metrics": metrics,
                        }
                    )

    averages = {name: _average(rows_by_condition[name]) for name in CONDITIONS}
    normal = averages["normal"]
    deltas = {
        name: {
            "delta_PSNR_vs_normal": averages[name]["PSNR"] - normal["PSNR"],
            "delta_SAM_vs_normal": averages[name]["SAM"] - normal["SAM"],
        }
        for name in CONDITIONS
    }

    test_conditions = {
        "method": adapter.canonical,
        "dataset": args.dataset,
        "mode": args.mode,
        "checkpoint": args.checkpoint,
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
        "evaluated_test_items": len(selected_indices),
        "selected_test_indices_1based": [i + 1 for i in selected_indices],
        "ablation_conditions": list(CONDITIONS),
        "pixel_shuffle": (
            "one shared spatial permutation across all HSI bands; each LR pixel spectrum preserved"
        ),
        "patch_shuffle": (
            "non-overlapping LR blocks permuted; within-block content preserved"
        ),
        "mean_broadcast": "per-band LR spatial mean broadcast to every LR position",
        "other_sample": (
            "another held-out test LR-HSI paired with current HR-MSI; when scene "
            "metadata is available, donor is chosen from a different held-out scene"
        ),
        "stochastic_control": (
            "identical reconstruction RNG seed across all six conditions for each sample/case"
        ),
        "seed": args.seed,
        "eval_seed": args.eval_seed,
        "model_details": details,
    }

    payload = {
        "test_conditions": test_conditions,
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

    print("=" * 108)
    print(
        f"LRHSI_DEPENDENCY model={adapter.canonical} dataset={args.dataset} "
        f"mode={args.mode} source={args.source_deformation}"
    )
    for name in CONDITIONS:
        m, d = averages[name], deltas[name]
        print(
            f"{name:14s} "
            f"PSNR={m['PSNR']:.6f} SAM={m['SAM']:.6f} "
            f"dPSNR={d['delta_PSNR_vs_normal']:+.6f} "
            f"dSAM={d['delta_SAM_vs_normal']:+.6f}"
        )
    print("=" * 108)
    print(f"LRHSI_DEPENDENCY_SUMMARY={csv_path}")


if __name__ == "__main__":
    main()
