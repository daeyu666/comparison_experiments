"""Unified root launcher for real-world Augsburg-2 center-heldout Wald x3 training."""
from __future__ import annotations

import argparse
import shlex
import subprocess

from experiment_protocol import (
    ROOT, build_real_train_command, real_protocol_summary, resolve_model,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified real-world Augsburg-2 center-heldout training launcher"
    )
    p.add_argument("--model", required=True, help="e.g. UAFL or EMR-Diff")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = resolve_model(args.model)
    if not adapter.real_train:
        raise ValueError(
            f"{adapter.canonical} has no registered real Augsburg train adapter"
        )
    cmd = build_real_train_command(adapter, args.device)
    print(f"REAL_AUGSBURG_TRAIN model={adapter.canonical}")
    print(f"PROTOCOL {real_protocol_summary()}")
    print("COMMAND " + shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
