"""Diagnose the Torch-2.6 PRFCoAM compatibility backend.

Run from repository root:
    python comparison/PRFCoAM/env_check.py

The adapted PRFCoAM no longer requires the author's legacy causal_conv1d CUDA
ABI. It uses native PyTorch depthwise conv1d plus the installed mamba_ssm
``selective_scan_fn`` CUDA backend.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys

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


def _test_selective_scan() -> bool:
    try:
        from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
    except Exception as exc:
        print("selective_scan_fn import: FAILED")
        print(f"  {type(exc).__name__}: {exc}")
        return False

    print("selective_scan_fn import: OK")
    if not torch.cuda.is_available():
        print("selective_scan CUDA smoke: SKIPPED (CUDA unavailable)")
        return False

    try:
        device = torch.device("cuda")
        b, d, n, l = 1, 4, 4, 8
        u = torch.randn(b, d, l, device=device, requires_grad=True)
        delta = torch.randn(b, d, l, device=device, requires_grad=True)
        A = -torch.arange(1, n + 1, device=device, dtype=torch.float32).repeat(d, 1)
        B = torch.randn(b, n, l, device=device, requires_grad=True)
        C = torch.randn(b, n, l, device=device, requires_grad=True)
        D = torch.ones(d, device=device)
        z = torch.randn(b, d, l, device=device, requires_grad=True)
        out = selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            D,
            z=z,
            delta_bias=torch.zeros(d, device=device),
            delta_softplus=True,
            return_last_state=False,
        )
        loss = out.float().square().mean()
        loss.backward()
        finite = bool(torch.isfinite(out).all() and torch.isfinite(u.grad).all())
        print(f"selective_scan CUDA forward/backward: {'OK' if finite else 'NON-FINITE'}")
        print(f"  output shape: {tuple(out.shape)}")
        return finite
    except Exception as exc:
        print("selective_scan CUDA forward/backward: FAILED")
        print(f"  {type(exc).__name__}: {exc}")
        return False


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

    print("\n=== compatibility backend ===")
    ok = _test_selective_scan()
    print("native PyTorch causal depthwise conv: used by comparison/PRFCoAM/mamba_compat.py")
    print(f"PRFCoAM backend status: {'READY' if ok else 'NOT READY'}")


if __name__ == "__main__":
    main()
