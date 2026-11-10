"""Thin interface around the released three-level PMI-RFCoNet."""
from __future__ import annotations

import torch
from torch import nn
from .upstream_model import SuperResolutionModel

ARCHITECTURE_ID = "pmi_rfconet_released_three_level_v1"
UPSTREAM_COMMIT = "b52e0c84291b917574034da9081c3fac404d34e5"
UPSTREAM_BLOB_SHA = "53258b0de2089a788bde134e715114e3291a37e1"


class PMIRFCoNet(nn.Module):
    def __init__(self, hsi_channels, msi_channels, *, mid_channels=64, num_fc=4, drop=0.0, scale_ratio=4):
        super().__init__()
        self.hsi_channels, self.msi_channels = int(hsi_channels), int(msi_channels)
        self.mid_channels, self.num_fc, self.drop = int(mid_channels), int(num_fc), float(drop)
        if min(self.hsi_channels, self.msi_channels, self.mid_channels, self.num_fc) < 1:
            raise ValueError("Channel counts and num_fc must be positive")
        if not 0 <= self.drop < 1:
            raise ValueError("drop must be in [0,1)")
        self.scale_ratio = int(scale_ratio)
        if self.scale_ratio not in (3, 4):
            raise ValueError("PMI-RFCoNet supports released x4 and Wald x3")
        self.net = SuperResolutionModel(
            drop=self.drop, in_channels=self.hsi_channels, out_channels=self.hsi_channels,
            mid_channels=self.mid_channels, factor=self.scale_ratio, num_fc=self.num_fc,
            msi_channels=self.msi_channels,
        )

    @property
    def model_args(self):
        return dict(mid_channels=self.mid_channels, num_fc=self.num_fc, drop=self.drop)

    def forward(self, lr_hsi, hr_msi):
        if lr_hsi.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("Require BCHW observations")
        if lr_hsi.shape[0] != hr_msi.shape[0]:
            raise ValueError("Observation batch mismatch")
        if lr_hsi.shape[1] != self.hsi_channels or hr_msi.shape[1] != self.msi_channels:
            raise ValueError("HSI/MSI channels differ from model configuration")
        if tuple(v*self.scale_ratio for v in lr_hsi.shape[-2:]) != tuple(hr_msi.shape[-2:]):
            raise ValueError(f"PMI-RFCoNet requires matching x{self.scale_ratio} observations")
        if min(lr_hsi.shape[-2:]) < 16 or any(v % 8 for v in lr_hsi.shape[-2:]):
            raise ValueError("LR spatial dimensions must be multiples of 8 and at least 16 (HR>=64)")
        return self.net(lr_hsi, hr_msi)


def build_model(info, args, device):
    return PMIRFCoNet(
        int(info["n_bands"]), int(info["n_select_bands"]),
        mid_channels=int(getattr(args, "mid_channels", 64)),
        num_fc=int(getattr(args, "num_fc", 4)), drop=float(getattr(args, "drop", 0.0)),
    ).to(device)
