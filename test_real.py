"""Unified root launcher for real-world Augsburg-2 center-heldout evaluation.

One command performs:
  1) reduced-resolution heldout reference test (PSNR/SAM/RMSE);
  2) native 10m heldout inference;
  3) QNR/Dlambda/Ds on the native heldout ROI;
  4) GeoTIFF output.
"""
from __future__ import annotations

import argparse
import shlex
import subprocess

from experiment_protocol import (
    ROOT,
    build_real_native_test_command,
    build_real_reference_test_command,
    real_protocol_summary,
    resolve_model,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Unified real-world Augsburg-2 center-heldout test launcher"
    )
    p.add_argument("--model", required=True, help="e.g. UAFL or EMR-Diff")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    adapter = resolve_model(args.model)
    if not adapter.real_train or not adapter.real_infer:
        raise ValueError(
            f"{adapter.canonical} does not have complete real Augsburg adapters"
        )

    rr = build_real_reference_test_command(adapter, args.device)
    native = build_real_native_test_command(adapter, args.device)

    print(f"REAL_AUGSBURG_TEST model={adapter.canonical}")
    print(f"PROTOCOL {real_protocol_summary()}")
    print("STEP1_RR_REFERENCE_TEST " + shlex.join(rr))
    print("STEP2_NATIVE_QNR " + shlex.join(native))
    if args.dry_run:
        return

    subprocess.run(rr, cwd=ROOT, check=True)
    subprocess.run(native, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
