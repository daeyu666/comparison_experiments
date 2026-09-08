"""Diagnose PRFCoAM's self-contained Torch backend.

Run from repository root:
    python comparison/PRFCoAM/env_check.py

The adapted PRFCoAM no longer imports mamba_ssm, causal_conv1d_cuda or
selective_scan_cuda.  It uses native PyTorch causal depthwise convolution plus
an equivalent parallel affine-prefix implementation of the selective scan.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from mamba_compat import selective_scan_torch


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


def _sequential_reference(u, delta, A, B, C, D, z, delta_bias):
    """Tiny exact reference used only to validate the parallel prefix math."""
    dtype_in = u.dtype
    u_f = u.float()
    delta_f = torch.nn.functional.softplus(delta.float() + delta_bias.float().unsqueeze(-1))
    A_f = A.float()
    B_f = B.float()
    C_f = C.float()

    state = torch.zeros(
        u.shape[0], u.shape[1], A.shape[1],
        device=u.device, dtype=torch.float32,
    )
    ys = []
    for i in range(u.shape[-1]):
        a_i = torch.exp(delta_f[:, :, i].unsqueeze(-1) * A_f.unsqueeze(0))
        b_i = (
            delta_f[:, :, i].unsqueeze(-1)
            * B_f[:, :, i].unsqueeze(1)
            * u_f[:, :, i].unsqueeze(-1)
        )
        state = a_i * state + b_i
        ys.append((state * C_f[:, :, i].unsqueeze(1)).sum(dim=-1))
    out = torch.stack(ys, dim=-1)
    out = out + u_f * D.float().view(1, -1, 1)
    out = out * torch.nn.functional.silu(z.float())
    return out.to(dtype_in)


def _test_self_contained_scan() -> bool:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        torch.manual_seed(123)
        b, d, n, l = 1, 4, 4, 9
        u = torch.randn(b, d, l, device=device, requires_grad=True)
        delta = torch.randn(b, d, l, device=device, requires_grad=True)
        A = -torch.arange(1, n + 1, device=device, dtype=torch.float32).repeat(d, 1)
        B = torch.randn(b, n, l, device=device, requires_grad=True)
        C = torch.randn(b, n, l, device=device, requires_grad=True)
        D = torch.ones(d, device=device)
        z = torch.randn(b, d, l, device=device, requires_grad=True)
        delta_bias = torch.zeros(d, device=device)

        out = selective_scan_torch(
            u, delta, A, B, C, D,
            z=z,
            delta_bias=delta_bias,
            delta_softplus=True,
            return_last_state=False,
        )
        ref = _sequential_reference(u, delta, A, B, C, D, z, delta_bias)
        max_err = float((out - ref).abs().max().detach().cpu())

        loss = out.float().square().mean()
        loss.backward()
        finite = bool(
            torch.isfinite(out).all()
            and u.grad is not None
            and torch.isfinite(u.grad).all()
        )
        close = max_err < 1e-4
        print(f"self-contained selective scan numerical check: {'OK' if close else 'FAILED'}")
        print(f"  max abs error vs sequential reference: {max_err:.3e}")
        print(f"self-contained selective scan forward/backward: {'OK' if finite else 'NON-FINITE'}")
        print(f"  device: {device}")
        print(f"  output shape: {tuple(out.shape)}")
        return finite and close
    except Exception as exc:
        print("self-contained selective scan: FAILED")
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
    print(f"installed causal-conv1d: {_pkg_version('causal-conv1d')} (ignored by PRFCoAM adapter)")
    print(f"installed mamba-ssm: {_pkg_version('mamba-ssm')} (ignored by PRFCoAM adapter)")

    nvcc = shutil.which("nvcc")
    print(f"nvcc path: {nvcc}")
    if nvcc:
        print("nvcc version:")
        print(_run([nvcc, "--version"]))
    print(f"CUDA_HOME env: {os.environ.get('CUDA_HOME', '<unset>')}")

    print("\n=== self-contained compatibility backend ===")
    ok = _test_self_contained_scan()
    print("native PyTorch causal depthwise conv: ENABLED")
    print("external Mamba CUDA extensions required: NO")
    print(f"PRFCoAM backend status: {'READY' if ok else 'NOT READY'}")


if __name__ == "__main__":
    main()
