"""Minimal adapter around the author's PRFCoAM implementation.

The original ``base/`` code is kept untouched. This adapter fixes assumptions
that are tied to the released PaviaC / old-Mamba environment:

* HSI channels hard-coded to 102 inside the custom Mamba block;
* MSI channels hard-coded to 4 inside the same block;
* module-level CUDA device globals used by the spatial transformer; and
* the released Mamba-1.0.1 fused CUDA ABI.

The PRFCoAM registration/fusion topology is not changed. The custom Mamba class
is provided by ``mamba_compat.py`` and uses only native PyTorch operators.
No installed ``mamba_ssm`` package or compiled Mamba CUDA extension is needed.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Tuple

import torch
from torch import nn

_THIS_DIR = Path(__file__).resolve().parent
_BASE_DIR = _THIS_DIR / "base"
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

# Load the self-contained compatibility block before base/ is added to sys.path.
# Then create the minimal module hierarchy expected by the author's import:
#   from mamba_ssm.modules.mamba_simple_4scan_xiugai import Mamba
# This deliberately avoids importing either an installed mamba_ssm package or
# the copied base/mamba_ssm package, both of which try to load incompatible .so
# files in the target Torch-2.6 environment.
try:
    import mamba_compat as official_mamba  # type: ignore

    mamba_pkg = types.ModuleType("mamba_ssm")
    mamba_pkg.__path__ = []  # mark as package
    modules_pkg = types.ModuleType("mamba_ssm.modules")
    modules_pkg.__path__ = []

    setattr(mamba_pkg, "modules", modules_pkg)
    setattr(modules_pkg, "mamba_simple_4scan_xiugai", official_mamba)

    sys.modules["mamba_ssm"] = mamba_pkg
    sys.modules["mamba_ssm.modules"] = modules_pkg
    sys.modules["mamba_ssm.modules.mamba_simple_4scan_xiugai"] = official_mamba
except Exception as exc:  # pragma: no cover - clearer local setup error
    raise RuntimeError(
        "Failed to initialize the self-contained PRFCoAM Mamba backend. "
        "The backend only requires PyTorch and einops; see "
        "comparison/PRFCoAM/README.md."
    ) from exc

if str(_BASE_DIR) not in sys.path:
    sys.path.insert(0, str(_BASE_DIR))

try:
    import model_ssm_fuse9_2 as official_model  # type: ignore
except Exception as exc:  # pragma: no cover - clearer local setup error
    raise RuntimeError(
        "Failed to import the author's PRFCoAM base implementation after "
        "injecting the self-contained Mamba backend. See "
        "comparison/PRFCoAM/README.md."
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
            module.Cin = int(hsi_channels)
            module.ca = official_mamba.ChannelAttentionModule(int(hsi_channels))
        elif module.bimamba_type == "v3":
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
