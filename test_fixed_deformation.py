"""Run the six fixed-rigid/random-local strengths on every held-out patch."""
from __future__ import annotations

import argparse
import importlib.util
import json
from dataclasses import asdict
from pathlib import Path

import torch

from experiment_protocol import DATASETS, MODES, ROOT, checkpoint_path, resolve_model
from fixed_deformation_protocol import STRENGTHS, add_suite_args, run_fixed_suite


def _load_evaluator(folder):
    path = ROOT / "comparison" / folder / "test_hsi_deformed.py"
    spec = importlib.util.spec_from_file_location(f"fixed_{folder.replace('-', '_')}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _emr(args):
    evaluator = _load_evaluator("EMR-Diff")
    trainer = evaluator.build_trainer(args)
    checkpoint = torch.load(args.checkpoint, map_location=trainer.device, weights_only=False)
    trainer.verify_checkpoint_architecture(checkpoint, context=f"fixed test {args.checkpoint}")
    if checkpoint.get("dataset") != args.dataset:
        raise ValueError("checkpoint/test dataset mismatch")
    if int(checkpoint.get("state_channels", -1)) != trainer.state_channels:
        raise ValueError("checkpoint/test state-channel mismatch")
    if checkpoint.get("degradation_mode") != "physical":
        raise ValueError("fixed test requires a physical-degradation checkpoint")
    stage = checkpoint.get("training_stage")
    if args.mode == "mixed" and stage != "stage2_registered_deformed_mixed":
        raise ValueError("--mode mixed requires the stage-2 mixed checkpoint")
    if args.mode == "registered" and stage not in (None, "stage1_registered"):
        raise ValueError("--mode registered requires the stage-1 checkpoint")
    trainer.Net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.Net.eval()
    test_set = trainer.test_dataloader.dataset
    p0 = test_set.degradation_operator.to(trainer.device)
    details = {"checkpoint_epoch": checkpoint.get("epoch"), "training_stage": stage,
               "architecture_id": checkpoint.get("architecture_id"),
               "registered_probability": checkpoint.get("registered_probability")}
    return test_set, trainer._reconstruct, p0.degrade, trainer.device, details


def _uafl(args):
    evaluator = _load_evaluator("UAFL")
    cfg = evaluator.build_shared_cfg(args)
    _, _, test_set, info = evaluator.build_datasets(cfg, include_validation=True)
    evaluator.require_srf_weights(info)
    device = torch.device(args.device)
    model = evaluator.build_uafl(int(info["n_select_bands"])).to(device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    train_args = checkpoint.get("args")
    if not isinstance(train_args, dict):
        raise ValueError("UAFL checkpoint must record training args for protocol verification")
    for name in ("dataset", "image_size", "patch_size", "stride", "scale_ratio",
                 "degradation_mode", "mtf_nyquist", "psf_truncate"):
        if name in train_args and train_args[name] != getattr(args, name):
            raise ValueError(f"UAFL checkpoint/test mismatch: {name}")
    if args.mode == "registered":
        if train_args.get("train_misalignment_mode") != "registered":
            raise ValueError("--mode registered requires the registered UAFL checkpoint")
    elif (train_args.get("train_misalignment_mode") is not None
          or "validation_cases" not in train_args or "max_local_px" not in train_args):
        raise ValueError("--mode mixed requires an HSI-deformed mixed checkpoint, not legacy MSI-warp weights")
    model.load_state_dict(checkpoint.get("model", checkpoint), strict=True)
    model.eval()
    p0 = test_set.degradation_operator.to(device)

    def predict(lr_hsi, hr_msi):
        return model(evaluator.upsample_lr_hsi(lr_hsi, hr_msi.shape[-2:]), hr_msi)

    details = {"checkpoint_epoch": checkpoint.get("epoch"),
               "training_stage": args.mode, "registered_probability": train_args.get("registered_probability")}
    return test_set, predict, p0.degrade, device, details


def _clsnet(args):
    from comparison.CLSNet.model import architecture_metadata
    from comparison.CLSNet.common import (
        build_shared_cfg, model_for_info, load_checkpoint, resolve_device,
    )
    from data_loader import build_datasets
    _, _, test_set, info = build_datasets(build_shared_cfg(args), include_validation=True)
    device = resolve_device(args.device)
    model = model_for_info(info, args, device)
    state = load_checkpoint(args.checkpoint, model, args, expected_stage=args.mode)
    model.eval()
    p0 = test_set.degradation_operator.to(device)
    details = {"checkpoint_epoch": state["epoch"], "training_stage": state["training_stage"],
               "registered_probability": state.get("registered_probability"),
               **architecture_metadata(model)}
    return test_set, model, p0.degrade, device, details


# Future comparison methods supply the same five values; the entire strength
# schedule, full-split traversal, local-field RNG and metrics stay shared.
MODEL_BUILDERS = {"EMR-Diff": _emr, "UAFL": _uafl, "CLSNet": _clsnet}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", choices=MODES, default="mixed",
                   help="Checkpoint training stage; both modes are tested at all selected strengths")
    p.add_argument("--checkpoint", default="")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--dry_run", action="store_true")
    add_suite_args(p)
    return p.parse_args(argv)


def main():
    args = parse_args()
    adapter = resolve_model(args.model)
    if adapter.canonical not in MODEL_BUILDERS:
        raise ValueError(f"Register a fixed-test model builder for {adapter.canonical}")
    args.checkpoint = str(Path(args.checkpoint).resolve() if args.checkpoint
                          else checkpoint_path(adapter, args.dataset, args.mode))
    args.data_root = str(Path(args.data_root).resolve())
    if not args.output_dir:
        args.output_dir = str(ROOT / "comparison" / adapter.folder / "outputs" / "fixed_deformation"
                              / args.dataset / f"{args.mode}_seed{args.seed}_cases{args.cases}_eval{args.eval_seed}")
    # The training/sensor protocol remains fixed; test strength never changes
    # network architecture or the geometry-estimator's training-time bounds.
    args.image_size, args.patch_size, args.stride, args.scale_ratio = 128, 64, 32, 4
    args.degradation_mode, args.mtf_nyquist, args.psf_truncate = "physical", 0.2, 3.0
    print(json.dumps({"model": adapter.canonical, "dataset": args.dataset, "mode": args.mode,
                      "checkpoint": args.checkpoint, "output_dir": args.output_dir,
                      "strengths": [asdict(STRENGTHS[i-1]) for i in args.levels]}, indent=2), flush=True)
    if args.cases < 1 or len(set(args.levels)) != len(args.levels):
        raise ValueError("cases must be >=1 and levels must be distinct")
    if args.dry_run:
        return
    if not Path(args.checkpoint).is_file():
        raise FileNotFoundError(args.checkpoint)
    if str(args.device).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; pass --device cpu for CPU testing")
    torch.manual_seed(args.seed)
    test_set, predict, degrade, device, details = MODEL_BUILDERS[adapter.canonical](args)
    run_fixed_suite(
        test_set, predict, degrade, device=device, output_dir=args.output_dir,
        metadata={"method": adapter.canonical, "dataset": args.dataset, "mode": args.mode,
                  "checkpoint": args.checkpoint, "data_root": args.data_root,
                  "degradation": "physical", "mtf_nyquist": 0.2, "psf_truncate": 3.0, **details},
        cases=args.cases, seed=args.seed, eval_seed=args.eval_seed, levels=args.levels,
        control_grid=args.control_grid, min_jacobian=args.min_jacobian,
    )


if __name__ == "__main__":
    main()
