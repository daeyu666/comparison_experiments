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
    p.add_argument("--model", required=True, help="e.g. UAFL, EMR-Diff, CLSNet, DRTNet, PSRF-DiffNet, PSRT, SA-RFM, PMI-RFCoNet, MoE-PNP or IR&ArF")
    p.add_argument("--dataset", required=True, choices=DATASETS)
    p.add_argument("--mode", required=True, choices=MODES)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=None,
                   help="EMR-Diff training batch override; default=2. Learning rate is unchanged.")
    p.add_argument("--lr", type=float, default=None,
                   help="Explicit Adam learning-rate override for IR&ArF or PMI-RFCoNet")
    p.add_argument("--grad_clip",type=float,default=None,help="PMI-RFCoNet only: gradient norm limit; 0 disables clipping")
    p.add_argument("--resume_optimizer_override",action="store_true",help="PMI-RFCoNet only: explicitly change lr/grad_clip on resume")
    p.add_argument("--repair_bn_stats",action="store_true",help="PMI-RFCoNet only: explicitly recalibrate damaged BN buffers using training observations")
    p.add_argument("--bn_calibration_batches",type=int,default=32)
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
                              epochs=args.epochs, resume=args.resume,
                              hidden_channels=args.hidden_channels, lr_override=args.lr,
                              grad_clip_override=args.grad_clip,resume_optimizer_override=args.resume_optimizer_override,
                              repair_bn_stats=args.repair_bn_stats,bn_calibration_batches=args.bn_calibration_batches)
    print(
        f"UNIFIED_TRAIN model={adapter.canonical} dataset={args.dataset} "
        f"mode={args.mode}"
    )
    print(f"PROTOCOL {protocol_summary(adapter, args.batch_size, epochs=args.epochs, mode=args.mode, hidden_channels=args.hidden_channels, lr_override=args.lr, grad_clip_override=args.grad_clip)}")
    print("COMMAND " + shlex.join(cmd))
    if args.dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
