"""Unified root training entry for all registered comparison adapters."""
from __future__ import annotations

import argparse
import shlex
import subprocess
from emr_configuration import add_hidden_channels_argument

from experiment_protocol import (
    DATASETS, MODES, ROOT, build_train_command, protocol_summary, resolve_model,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified two-stage comparison training launcher"
    )
    p.add_argument("--model", required=True, help="e.g. UAFL, EMR-Diff or CLSNet")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=None,
                   help="EMR-Diff training batch override; default=2. Learning rate is unchanged.")
    p.add_argument("--epochs", type=int, default=None,
                   help="Total target epochs, including completed epochs when resuming")
    p.add_argument("--resume", default="",
                   help="Checkpoint path, or 'auto' to use the latest checkpoint of this model/dataset/stage")
    p.add_argument("--dry_run", action="store_true")
    add_hidden_channels_argument(p)
    return p.parse_args()


def main():
    args = parse_args()
    adapter = resolve_model(args.model)
    cmd = build_train_command(adapter, args.dataset, args.mode, args.device, args.batch_size,
                              epochs=args.epochs, resume=args.resume, hidden_channels=args.hidden_channels)
    print(
        f"UNIFIED_TRAIN model={adapter.canonical} dataset={args.dataset} "
        f"mode={args.mode}"
    )
    print(f"PROTOCOL {protocol_summary(adapter, args.batch_size, epochs=args.epochs, mode=args.mode, hidden_channels=args.hidden_channels)}")
    print("COMMAND " + shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
