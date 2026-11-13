"""Train one method on the real Augsburg Region 2 center-heldout Wald x3 split."""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex
import subprocess

from emr_configuration import validate_hidden_channels
from experiment_protocol import ROOT, build_real_train_command, real_protocol_summary, resolve_model


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, help="EMR-Diff batch override")
    p.add_argument("--base_channels", "--base_channel", "--hidden_channels", "--model_width",
                   dest="hidden_channels", type=validate_hidden_channels, help="EMR-Diff only; default 64")
    p.add_argument("--epochs", type=int)
    p.add_argument("--resume", help="Local checkpoint path or auto")
    p.add_argument("--wald_root", type=Path)
    p.add_argument("--radiometry_json", type=Path)
    for key in ("checkpoint_dir", "log_dir"):
        p.add_argument("--" + key, type=lambda path: Path(path).expanduser().resolve())
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    adapter = resolve_model(args.model)
    resume = args.resume
    if resume and resume != "auto":
        resume = Path(resume).expanduser().resolve()
    cmd = build_real_train_command(
        adapter, args.device, args.batch_size, hidden_channels=args.hidden_channels,
        epochs=args.epochs, resume=resume, wald_root=args.wald_root, radiometry_json=args.radiometry_json,
        checkpoint_directory=args.checkpoint_dir, log_directory=args.log_dir,
    )
    print(f"REAL_AUGSBURG_TRAIN model={adapter.canonical}")
    print(f"PROTOCOL {real_protocol_summary(adapter, args.batch_size, hidden_channels=args.hidden_channels, epochs=args.epochs)}")
    print("COMMAND " + shlex.join(cmd), flush=True)
    if not args.dry_run:
        subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
