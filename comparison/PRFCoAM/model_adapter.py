"""Minimal adapter around the author's PRFCoAM implementation.

The original `base/` code is kept untouched. This adapter only fixes assumptions
that are tied to the author's PaviaC setup:
  * HSI channels hard-coded to 102 inside the custom Mamba block;
  * MSI channels hard-coded to 4 inside the same block;
  * module-level CUDA device globals used by the spatial transformer.

It does not change the PRFCoAM registration/fusion topology.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR / "base"
if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

try:
    import model_ssm_fuse9_2 as official_model  # type: ignore
    from mamba_ssm.modules import mamba_simple_4scan_xiugai as official_mamba  # type: ignore
except Exception as exc:  # pragma: no cover - gives a clearer local setup error
    raise RuntimeError(
        "Failed to import the author's PRFCoAM base implementation. "
        "Its bundled Mamba-1.0.1 Python code requires ABI-compatible "
        "causal_conv1d_cuda and selective_scan_cuda extensions. "
        "Run `python comparison/PRFCoAM/env_check.py` and see "
        "comparison/PRFCoAM/README.md for the compatibility build."
    ) from exc


def _make_relu_out_of_place(model: nn.Module) -> None:
    """Preserve ReLU math while avoiding accidental in-place autograd conflicts."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False


def _patch_mamba_channels(model: nn.Module, hsi_channels: int, msi_channels: int) -> int:
    patched = 0
    for module in model.modules():
        if not isinstance(module, official_mamba.Mamba):
            continue
        patched += 1
        if module.bimamba_type == "v2":
            # v2 scans 2x2 spatial patches along the spectral dimension. The
            # author's ChannelAttentionModule was constructed for exactly 102 bands.
            module.Cin = int(hsi_channels)
            module.ca = official_mamba.ChannelAttentionModule(int(hsi_channels))
        elif module.bimamba_type == "v3":
            # v3 is the spatial MSI scan and reshapes its output with Cout.
            module.Cout = int(msi_channels)
            module.sa = official_mamba.SpatialAttentionModule()
    return patched


def build_prfcoam(
    hsi_channels: int,
    msi_channels: int,
    device: torch.device,
) -> nn.Module:
    """Build the author's PRFCoAM topology with dataset-dynamic channels."""
    if int(hsi_channels) < 1 or int(msi_channels) < 1:
        raise ValueError("channel counts must be positive")

    # SpatialTransformation in the author code reads this module-global variable
    # every forward pass. Point it at the actual benchmark device rather than
    # the hard-coded cuda:1 used in the released code.
    official_model.device = device
    official_mamba.device = device

    model = official_model.Net(int(hsi_channels), int(msi_channels))
    patched = _patch_mamba_channels(model, int(hsi_channels), int(msi_channels))
    if patched < 1:
        raise RuntimeError("No PRFCoAM custom Mamba blocks were found to patch")
    _make_relu_out_of_place(model)
    model = model.to(device)
    return model


def displacement_smoothness(flow: torch.Tensor) -> torch.Tensor:
    """Same squared spatial smoothness penalty used by the released train.py."""
    if flow.ndim != 4:
        raise ValueError(f"expected BxCxHxW flow, got {tuple(flow.shape)}")
    dy = flow[:, :, 1:, :] - flow[:, :, :-1, :]
    dx = flow[:, :, :, 1:] - flow[:, :, :, :-1]
    return 0.5 * (dx.square().mean() + dy.square().mean())


def unpack_outputs(outputs) -> Tuple[torch.Tensor, ...]:
    if not isinstance(outputs, (tuple, list)) or len(outputs) != 6:
        raise RuntimeError("PRFCoAM Net is expected to return exactly 6 tensors")
    return tuple(outputs)


__all__ = ["build_prfcoam", "displacement_smoothness", "unpack_outputs"]
