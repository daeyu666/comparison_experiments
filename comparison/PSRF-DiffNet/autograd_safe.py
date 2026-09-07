"""Autograd-safety helpers for the PSRF-DiffNet reproduction.

The released-style fusion stack contains nested ReLU activations. When an
outer ``nn.ReLU(inplace=True)`` follows a residual block whose output was
already produced by an in-place functional ReLU, PyTorch can detect that a
saved tensor was modified before backward. This helper keeps the exact network
math while forcing all ReLU activations to be out-of-place.
"""

from __future__ import annotations

import types

import torch
from torch import nn

from network import ResidualBlock


def _safe_residual_forward(self: ResidualBlock, x: torch.Tensor) -> torch.Tensor:
    """ResidualBlock forward with an out-of-place final activation."""
    return torch.relu(self.body(x) + self.skip(x))


def make_autograd_safe(model: nn.Module) -> nn.Module:
    """Disable all in-place ReLUs and patch residual final ReLU safely."""
    for module in model.modules():
        if isinstance(module, nn.ReLU):
            module.inplace = False
        if isinstance(module, ResidualBlock):
            module.forward = types.MethodType(_safe_residual_forward, module)
    return model


__all__ = ["make_autograd_safe"]
