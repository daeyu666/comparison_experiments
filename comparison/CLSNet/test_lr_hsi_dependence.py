"""Diagnose whether a trained CLSNet relies on LR-HSI spatial correspondence.

The HR-MSI and target HSI are held fixed. Only the LR-HSI input is intervened:
normal registered LR-HSI, spatial pixel shuffle, LR-patch shuffle, spatial-mean
broadcast, zeros, and a cross-sample LR-HSI. Pixel/patch shuffles preserve the
LR-HSI spectral values while destroying their spatial correspondence.

This is deliberately an evaluation-only diagnostic. Retraining here would mix
the intervention into the learned model and would no longer answer whether an
existing registered/mixed checkpoint has learned to suppress LR-HSI spatial use.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from comparison.CLSNet.common import (
    HERE, add_data_args, add_geometry_args, build_shared_cfg, load_checkpoint,
    model_for_info, resolve_device,
)
from comparison.CLSNet.final_metrics import (
    average_metrics, calc_metrics, format_metrics,
)
from data_loader import build_datasets


POLICIES = (
    "normal",
    "spatial_shuffle",
    "patch_shuffle",
    "mean_broadcast",
    "zero",
    "cross_sample",
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--mode", choices=("registered", "mixed"), default="mixed",
                   help="Training stage of the checkpoint to diagnose.")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--patch_lr", type=int, default=4,
                   help="Non-overlapping LR-HSI patch size for patch_shuffle.")
    p.add_argument("--cross_offset", type=int, default=1,
                   help="Test-sample offset used by cross_sample.")
    p.add_argument("--policies", nargs="+", choices=POLICIES, default=list(POLICIES))
    p.add_argument("--output_json", default="")
    p.add_argument("--output_csv", default="")
    return p.parse_args(argv)


def _spatial_shuffle(lr, *, seed):
    b, c, h, w = lr.shape
    if b != 1:
        raise ValueError("diagnostic expects batch size 1")
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(h * w, generator=gen).to(lr.device)
    flat = lr.reshape(b, c, h * w)
    return flat.index_select(2, perm).reshape_as(lr)


def _patch_shuffle(lr, *, patch, seed):
    b, c, h, w = lr.shape
    if b != 1:
        raise ValueError("diagnostic expects batch size 1")
    patch = int(patch)
    if patch < 1 or h % patch or w % patch:
        raise ValueError(
            f"--patch_lr must divide the LR-HSI size exactly; got patch={patch}, size={h}x{w}"
        )
    gh, gw = h // patch, w // patch
    blocks = (
        lr.reshape(b, c, gh, patch, gw, patch)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(b, c, gh * gw, patch, patch)
    )
    gen = torch.Generator(device="cpu").manual_seed(int(seed))
    perm = torch.randperm(gh * gw, generator=gen).to(lr.device)
    blocks = blocks.index_select(2, perm)
    return (
        blocks.reshape(b, c, gh, gw, patch, patch)
        .permute(0, 1, 2, 4, 3, 5)
        .reshape(b, c, h, w)
    )


def _intervene(policy, lr, *, sample_index, test_set, p0, device, patch_lr, cross_offset, seed):
    if policy == "normal":
        return lr
    if policy == "spatial_shuffle":
        return _spatial_shuffle(lr, seed=seed + 100003 * sample_index + 17)
    if policy == "patch_shuffle":
        return _patch_shuffle(
            lr, patch=patch_lr, seed=seed + 100003 * sample_index + 29
        )
    if policy == "mean_broadcast":
        return lr.mean(dim=(-2, -1), keepdim=True).expand_as(lr)
    if policy == "zero":
        return torch.zeros_like(lr)
    if policy == "cross_sample":
        if len(test_set) < 2:
            raise ValueError("cross_sample requires at least two test samples")
        offset = int(cross_offset) % len(test_set)
        if offset == 0:
            raise ValueError("--cross_offset must not be a multiple of the test-set size")
        other = test_set[(sample_index + offset) % len(test_set)]
        other_gt = other["gt"].unsqueeze(0).to(device=device, dtype=lr.dtype)
        other_lr = p0.degrade(other_gt)
        if other_lr.shape != lr.shape:
            raise ValueError(
                f"cross-sample LR-HSI shape mismatch: {tuple(other_lr.shape)} vs {tuple(lr.shape)}"
            )
        return other_lr
    raise ValueError(f"unsupported policy: {policy}")


def main(argv=None):
    args = parse_args(argv)
    if args.patch_lr < 1:
        raise ValueError("--patch_lr must be >=1")
    if not args.policies or len(set(args.policies)) != len(args.policies):
        raise ValueError("--policies must be non-empty and distinct")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = resolve_device(args.device)

    _, _, test_set, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if len(test_set) < 1:
        raise ValueError("empty CLSNet test split")
    p0 = test_set.degradation_operator.to(device)
    model = model_for_info(info, args, device)

    family = args.degradation_mode if args.mode == "registered" else "hsi_warp_final"
    ckpt = (
        Path(args.checkpoint)
        if args.checkpoint
        else HERE / "checkpoints" / family / args.dataset / "best.pth.tar"
    )
    state = load_checkpoint(ckpt, model, args, expected_stage=args.mode)
    model.eval()

    rows = {policy: [] for policy in args.policies}
    with torch.no_grad():
        for sample_index in range(len(test_set)):
            item = test_set[sample_index]
            gt = item["gt"].unsqueeze(0).to(device=device, dtype=torch.float32)
            hr_msi = item["hr_msi"].unsqueeze(0).to(device=device, dtype=torch.float32)
            lr = p0.degrade(gt)
            for policy in args.policies:
                lr_used = _intervene(
                    policy, lr,
                    sample_index=sample_index,
                    test_set=test_set,
                    p0=p0,
                    device=device,
                    patch_lr=args.patch_lr,
                    cross_offset=args.cross_offset,
                    seed=args.seed,
                )
                pred = model(lr_used, hr_msi)
                rows[policy].append(calc_metrics(pred, gt, args.scale_ratio))

    averages = {policy: average_metrics(values) for policy, values in rows.items()}
    baseline = averages.get("normal")
    summary_rows = []
    for policy in args.policies:
        metrics = averages[policy]
        row = {"policy": policy, **metrics}
        if baseline is not None:
            row["delta_PSNR_vs_normal"] = metrics["PSNR"] - baseline["PSNR"]
            row["delta_SAM_vs_normal"] = metrics["SAM"] - baseline["SAM"]
        summary_rows.append(row)

    conditions = {
        "method": "CLSNet",
        "dataset": args.dataset,
        "checkpoint": str(ckpt),
        "checkpoint_epoch": state.get("epoch"),
        "checkpoint_training_stage": state.get("training_stage"),
        "checkpoint_registered_probability": state.get("registered_probability"),
        "diagnostic": "LR-HSI input dependence",
        "base_lr_hsi": "registered P0(X)",
        "hr_msi": "unchanged R0(X)",
        "target": "unchanged X",
        "spatial_shuffle": "one shared spatial permutation across all HSI bands; exact LR spectra preserved",
        "patch_shuffle": f"non-overlapping {args.patch_lr}x{args.patch_lr} LR patches shuffled; within-patch spectra preserved",
        "mean_broadcast": "per-band spatial mean broadcast to every LR pixel",
        "zero": "all LR-HSI values set to zero",
        "cross_sample": f"registered LR-HSI from test sample +{args.cross_offset}",
        "test_patch_count": len(test_set),
        "seed": args.seed,
    }
    payload = {
        "test_conditions": conditions,
        "average_metrics": averages,
        "summary": summary_rows,
    }

    print("=" * 118)
    print("CLSNET_LR_HSI_DEPENDENCE_DIAGNOSTIC")
    print(
        f"dataset={args.dataset} mode={args.mode} epoch={state.get('epoch')} checkpoint={ckpt}"
    )
    for row in summary_rows:
        delta = ""
        if "delta_PSNR_vs_normal" in row:
            delta = (
                f" dPSNR={row['delta_PSNR_vs_normal']:+.6f}"
                f" dSAM={row['delta_SAM_vs_normal']:+.6f}"
            )
        print(f"{row['policy']:<16} {format_metrics(averages[row['policy']])}{delta}")
    print("=" * 118)

    tag = f"{args.mode}_seed{args.seed}"
    out_json = (
        Path(args.output_json)
        if args.output_json
        else HERE / "outputs" / "lr_hsi_dependence" / args.dataset / f"{tag}.json"
    )
    out_csv = (
        Path(args.output_csv)
        if args.output_csv
        else HERE / "outputs" / "lr_hsi_dependence" / args.dataset / f"{tag}.csv"
    )
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    if summary_rows:
        with out_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary_rows[0].keys()))
            writer.writeheader()
            writer.writerows(summary_rows)
    print(f"DIAGNOSTIC_JSON={out_json}")
    print(f"DIAGNOSTIC_CSV={out_csv}")


if __name__ == "__main__":
    main()
