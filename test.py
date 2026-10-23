"""Unified root final-test entry for all registered comparison adapters."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from emr_configuration import add_hidden_channels_argument

from experiment_protocol import (
    DATASETS, MODES, ROOT, TEST_GEOMETRY_SEEDS, build_test_command, protocol_summary, resolve_model,
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Unified two-stage comparison final-test launcher"
    )
    p.add_argument("--model", required=True, help="e.g. UAFL, EMR-Diff, CLSNet, DRTNet, PSRF-DiffNet or MoE-PNP")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--checkpoint", default="")
    add_hidden_channels_argument(p)
    p.add_argument("--geometry_seeds", type=int, nargs="+", default=list(TEST_GEOMETRY_SEEDS))
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--output_dir", default="")
    p.add_argument("--max_samples", type=int, default=0)
    p.add_argument("--resume_output", action="store_true")
    p.add_argument("--overwrite_output", action="store_true")
    p.add_argument("--legacy_single_seed", action="store_true", help="Reproduce the old final-test entry instead of new mixed-model 10x10 evaluation")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    if args.resume_output and args.overwrite_output:
        p.error("Choose resume_output OR overwrite_output")
    if (args.mode == "registered" or args.legacy_single_seed) and (args.resume_output or args.overwrite_output or args.max_samples):
        p.error("resume_output/overwrite_output/max_samples apply only to the mixed multiseed test")
    return args


def main(argv=None):
    args = parse_args(argv)
    adapter = resolve_model(args.model)
    cmd = build_test_command(adapter, args.dataset, args.mode, args.device,
        geometry_seeds=args.geometry_seeds, cases=args.cases, eval_seed=args.eval_seed,
        data_root=args.data_root, checkpoint_override=args.checkpoint, output_override=args.output_dir,
        max_samples=args.max_samples, resume_output=args.resume_output,
        overwrite_output=args.overwrite_output, legacy_single_seed=args.legacy_single_seed,
        check_files=not args.dry_run, hidden_channels=args.hidden_channels)
    print(
        f"UNIFIED_TEST model={adapter.canonical} dataset={args.dataset} "
        f"mode={args.mode}"
    )
    print(f"PROTOCOL {protocol_summary(adapter, hidden_channels=args.hidden_channels)}")
    if args.mode == "mixed" and not args.legacy_single_seed:
        print(f"MAIN_TABLE_TEST geometry_seeds={args.geometry_seeds} cases_per_patch_per_seed={args.cases}; Registered once; Warp mean+/-sample_std over seed means")
    print("COMMAND " + shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
