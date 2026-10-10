"""Unified root training entry for all registered comparison adapters."""
from __future__ import annotations

import argparse
import shlex
import subprocess

from experiment_protocol import (
    DATASETS, MODES, ROOT, build_train_command, protocol_summary, resolve_model,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified two-stage comparison training launcher"
    )
    p.add_argument("--model", required=True, help="e.g. UAFL or EMR-Diff")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = resolve_model(args.model)
    cmd = build_train_command(adapter, args.dataset, args.mode, args.device)
    print(
        f"UNIFIED_TRAIN model={adapter.canonical} dataset={args.dataset} "
        f"mode={args.mode}"
    )
    print(f"PROTOCOL {protocol_summary()}")
    print("COMMAND " + shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
