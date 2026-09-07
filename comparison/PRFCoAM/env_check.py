"""Diagnose PRFCoAM Mamba/CUDA extension compatibility.

Run from the repository root:
    python comparison/PRFCoAM/env_check.py

This script intentionally imports the compiled CUDA extensions directly so ABI
mismatches are reported before PRFCoAM itself is imported.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch


def _pkg_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def _run(cmd: list[str]) -> str:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True).strip()
    except Exception as exc:
        return f"unavailable ({exc})"


def _import_extension(name: str) -> None:
    try:
        mod = importlib.import_module(name)
        print(f"{name}: IMPORT OK")
        print(f"  file: {getattr(mod, '__file__', '<built-in>')}")
    except Exception as exc:
        print(f"{name}: IMPORT FAILED")
        print(f"  {type(exc).__name__}: {exc}")


def main() -> None:
    print("=== PRFCoAM environment diagnosis ===")
    print(f"python: {sys.version.split()[0]}")
    print(f"platform: {platform.platform()}")
    print(f"torch: {torch.__version__}")
    print(f"torch CUDA runtime: {torch.version.cuda}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU capability: {torch.cuda.get_device_capability(0)}")
    abi = getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", None)
    print(f"torch CXX11 ABI: {abi}")
    print(f"causal-conv1d package: {_pkg_version('causal-conv1d')}")
    print(f"mamba-ssm package: {_pkg_version('mamba-ssm')}")

    nvcc = shutil.which("nvcc")
    print(f"nvcc path: {nvcc}")
    if nvcc:
        print("nvcc version:")
        print(_run([nvcc, "--version"]))
    print(f"CUDA_HOME env: {os.environ.get('CUDA_HOME', '<unset>')}")

    print("\n=== compiled extension imports ===")
    _import_extension("causal_conv1d_cuda")
    _import_extension("selective_scan_cuda")

    print("\n=== author code expectation ===")
    print("bundled mamba_ssm __version__: 1.0.1")
    print("bundled causal wrapper expects legacy 4-argument causal_conv1d_fwd API")
    print("recommended compatibility target: mamba-ssm 1.0.1 + causal-conv1d 1.0.2")
    print("build both against the currently imported torch with --no-build-isolation")


if __name__ == "__main__":
    main()
