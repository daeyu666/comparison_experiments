"""Base interface for switchable HSI degradation operators."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class BaseDegradation(nn.Module, ABC):
    """Common interface used by ordinary and physical degradation modes."""

    mode: str = "base"

    def __init__(self, scale_ratio: int = 4):
        super().__init__()
        if scale_ratio < 1:
            raise ValueError("scale_ratio must be >= 1")
        self.scale_ratio = int(scale_ratio)

    def degrade(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the terminal observation operator."""
        return self.degrade_at(x, scale=self.scale_ratio, strength=1.0)

    @abstractmethod
    def degrade_at(
        self, x: torch.Tensor, *, scale: int, strength: float
    ) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"mode={self.mode}, scale_ratio={self.scale_ratio}"
