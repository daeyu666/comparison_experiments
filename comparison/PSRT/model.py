"""PSRT HSI/MSI wrapper for the shared comparison protocol.

The released PSRTnet receives (HR-MSI, bicubic-upsampled LR-HSI), and returns
a learned residual added to the upsampled HSI. The registered comparison uses
the same input convention and the released 2x3 pyramid shuffle Transformer.
"""
from __future__ import annotations

from types import SimpleNamespace
import torch
from torch import nn
import torch.nn.functional as F

from model_SR import PSRTnet

ARCHITECTURE_ID = "PSRT-TGRS2023-official-2x3-W2W-dynamic-channels-v1"
UPSTREAM_SOURCE_SHA = "07c41e4290719edc2fc1fd15ef56fec69d662d42"


class PSRTFusion(nn.Module):
    def __init__(self, hsi_channels: int, msi_channels: int, *, patch_size: int = 64):
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.patch_size = int(patch_size)
        if self.patch_size < 8 or self.patch_size % 8:
            raise ValueError("PSRT's 8/4/2 window pyramid requires a multiple-of-8 patch")
        self.net = PSRTnet(SimpleNamespace(
            n_bands=self.hsi_channels,
            n_bands_rgb=self.msi_channels,
            patch_size=self.patch_size,
        ))

    def forward(self, lr_hsi: torch.Tensor, hr_msi: torch.Tensor):
        if lr_hsi.ndim != 4 or hr_msi.ndim != 4:
            raise ValueError("PSRT expects BCHW tensors")
        if lr_hsi.shape[0] != hr_msi.shape[0]:
            raise ValueError("PSRT HSI/MSI batch sizes differ")
        if lr_hsi.shape[1] != self.hsi_channels or hr_msi.shape[1] != self.msi_channels:
            raise ValueError("PSRT HSI/MSI band counts differ from model configuration")
        h, w = hr_msi.shape[-2:]
        if h % 8 or w % 8:
            raise ValueError("PSRT requires HR height and width divisible by 8")
        lms = F.interpolate(lr_hsi, size=(h, w), mode="bicubic", align_corners=False)
        return self.net(hr_msi, lms)


def build_model(info, args, device):
    return PSRTFusion(
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        patch_size=int(getattr(args, "patch_size", 64)),
    ).to(device)
