"""Shared main-table mixed-checkpoint test, selectable by --model and --dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from experiment_protocol import DATASETS, ROOT, checkpoint_path, resolve_model, model_hidden_channels
from emr_configuration import add_hidden_channels_argument, variant_suffix
from test_fixed_deformation import MODEL_BUILDERS
from multiseed_deformation_protocol import (
    PROTOCOL_ID, add_multiseed_args, file_sha256, run_multiseed_suite, validate_settings,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--model", required=True,
        help=("Mixed-trained model adapter: UAFL, EMR-Diff, CLSNet, DRTNet, PSRF-DiffNet, or MoE-PNP; "
              "model names are case-insensitive and registered aliases are accepted."),
    )
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", choices=["mixed"], default="mixed")
    p.add_argument("--checkpoint", default="")
    add_hidden_channels_argument(p)
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10, help="Model/data setup seed; geometry seeds are independent")
    p.add_argument("--dry_run", action="store_true")
    add_multiseed_args(p)
    # Match existing per-method test defaults, including CLSNet provenance checks.
    p.set_defaults(image_size=128, patch_size=64, stride=32, scale_ratio=4,
                   degradation_mode="physical", mtf_nyquist=0.2, psf_truncate=3.0,
                   max_translation=4.0, max_rotation_deg=2.0, max_local_px=4.0,
                   control_grid=5, min_jacobian=0.5, registered_probability=0.10,
                   degradation_sigma=2.0, degradation_kernel_size=5, msi_mode="srf",
                   srf_path="", wavelength_root="./data/wavelengths", wavelength_path="",
                   srf_interp="pchip", srf_band_set="auto")
    args = p.parse_args(argv)
    validate_settings(args.geometry_seeds, args.cases, args.max_samples)
    if args.resume_output and args.overwrite_output:
        p.error("Choose resume_output OR overwrite_output")
    return args


def main(argv=None):
    args = parse_args(argv)
    adapter = resolve_model(args.model)
    model_hidden_channels(adapter, args.hidden_channels)
    if adapter.canonical not in MODEL_BUILDERS:
        raise ValueError(f"Register a shared test model builder for {adapter.canonical} in test_fixed_deformation.MODEL_BUILDERS")
    args.checkpoint = str(Path(args.checkpoint).resolve() if args.checkpoint
                          else checkpoint_path(adapter, args.dataset, "mixed", args.hidden_channels))
    args.data_root = str(Path(args.data_root).resolve())
    if not args.output_dir:
        tag = hashlib.sha256(",".join(map(str, args.geometry_seeds)).encode()).hexdigest()[:8]
        name = f"mixed_{len(args.geometry_seeds)}seeds_{args.cases}cases_{tag}_eval{args.eval_seed}{variant_suffix(args.hidden_channels)}"
        if args.max_samples:
            name += f"_SMOKE{args.max_samples}"
        args.output_dir = str(ROOT / "comparison" / adapter.folder / "outputs" / "multiseed_deformation" / args.dataset / name)
    print(json.dumps({"method": adapter.canonical, "protocol": PROTOCOL_ID, **vars(args)}, indent=2), flush=True)
    if args.dry_run:
        return
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu explicitly")
    torch.manual_seed(args.seed)
    test_set, predict, degrade, device, details = MODEL_BUILDERS[adapter.canonical](args)
    return run_multiseed_suite(
        test_set, predict, degrade, device=device, output_dir=args.output_dir,
        metadata={"method": adapter.canonical, "dataset": args.dataset, "mode": "mixed",
                  "checkpoint": args.checkpoint, "checkpoint_sha256": file_sha256(args.checkpoint),
                  "data_root": args.data_root, "degradation": "physical", "mtf_nyquist": 0.2,
                  "psf_truncate": 3.0, **details},
        geometry_seeds=args.geometry_seeds, cases=args.cases, eval_seed=args.eval_seed,
        control_grid=args.control_grid, min_jacobian=args.min_jacobian,
        max_translation=args.max_translation, max_rotation_deg=args.max_rotation_deg,
        max_local_px=args.max_local_px, scale_ratio=args.scale_ratio,
        max_samples=args.max_samples, resume_output=args.resume_output,
        overwrite_output=args.overwrite_output,
    )


if __name__ == "__main__":
    main()
