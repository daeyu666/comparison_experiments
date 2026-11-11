"""MoG-DCN, independently implemented from Dong et al. (TIP 2021)
and the architecture used by the authors' CAVE training entry.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_ID = "mog-dcn-cave-learned-operators-v1"
UPSTREAM_COMMIT = "35eef37dbef797739212cc3b65f747a432a28892"


def conv(i, o, k=3, stride=1):
    return nn.Conv2d(i, o, k, stride=stride, padding=k // 2)


class AnalysisLevel(nn.Module):
    def __init__(self, *, terminal=False):
        super().__init__()
        self.layers = nn.ModuleList(conv(64, 64) for _ in range(3))
        self.skip = conv(64, 128)
        self.reduce = None if terminal else conv(128, 64, stride=2)
        self.activation = nn.PReLU()

    def forward(self, x):
        for layer in self.layers:
            x = self.activation(layer(x))
        skip = self.skip(x)
        return skip, None if self.reduce is None else self.activation(self.reduce(skip))


class SynthesisLevel(nn.Module):
    def __init__(self, incoming, outgoing, output_kernel):
        super().__init__()
        self.expand = nn.ConvTranspose2d(incoming, 128, 3, stride=2, padding=1)
        self.layers = nn.ModuleList([conv(256, 128), conv(128, 128), conv(128, 128)])
        self.project = conv(128, outgoing, output_kernel)
        self.activation = nn.PReLU()

    def forward(self, x, skip):
        x = torch.cat((self.expand(x, output_size=skip.shape[-2:]), skip), dim=1)
        for layer in self.layers:
            x = self.activation(layer(x))
        return self.project(x)


class SharedDenoiser(nn.Module):
    def __init__(self, hsi_channels):
        super().__init__()
        self.encoder = nn.ModuleList(AnalysisLevel() for _ in range(4))
        self.bridge = AnalysisLevel(terminal=True)
        self.decoder = nn.ModuleList(
            SynthesisLevel(i, o, k)
            for i, o, k in ((128, 512, 1), (512, 512, 1), (512, 512, 1), (512, hsi_channels, 3))
        )

    def forward(self, x):
        skips = []
        for layer in self.encoder:
            skip, x = layer(x)
            skips.append(skip)
        x, _ = self.bridge(x)
        for layer, skip in zip(self.decoder, reversed(skips)):
            x = layer(x, skip)
        return x, skips[0]


class MoGDCN(nn.Module):
    """Four unfolding stages, each using the same denoiser twice."""

    def __init__(self, hsi_channels: int, msi_channels: int, scale: int = 4):
        super().__init__()
        self.scale = scale
        self.hsi_channels, self.msi_channels = hsi_channels, msi_channels
        self.denoise = SharedDenoiser(hsi_channels)
        self.embed = conv(hsi_channels, 64)
        self.memory = nn.ModuleList(conv(64 + 128 * n, 64) for n in range(1, 4))
        self.refine = nn.ModuleList(conv(192, 64) for _ in range(4))
        self.spectral_forward = conv(hsi_channels, msi_channels)
        self.spectral_back = conv(msi_channels, hsi_channels)
        # Keep the released 13x13 dense learned operators; change the spatial
        # stride from eight to the benchmark ratio. These are internal model
        # operators, separate from the fixed PSF/SRF used to generate inputs.
        self.spatial_forward = conv(hsi_channels, hsi_channels, 13, stride=scale)
        self.spatial_back = nn.ConvTranspose2d(hsi_channels, hsi_channels, 13, stride=scale, padding=6)
        self.prior_step = nn.Parameter(torch.tensor(0.1))
        self.prior_penalty = nn.Parameter(torch.tensor(0.9))
        self.data_step = nn.Parameter(torch.tensor(0.1))
        self.data_penalty = nn.Parameter(torch.tensor(0.9))
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_normal_(module.weight)

    def spectral_residual(self, estimate, msi):
        return self.spectral_back(msi - self.spectral_forward(estimate))

    def forward(self, lr_hsi, hr_msi, *, observer=None):
        """Optionally expose intermediate tensors for read-only diagnostics."""
        x = F.interpolate(lr_hsi, scale_factor=self.scale, mode="bicubic", align_corners=False)
        if observer is not None:
            observer(0, "initial", x)
        memory = []
        for stage, project in enumerate(self.refine):
            features = self.embed(x)
            if stage:
                features = self.memory[stage - 1](torch.cat((features, *memory), dim=1))
            noise, skip = self.denoise(features)
            memory.append(skip)
            # Released CAVE path: same estimate is used for the noisy anchor
            # and the current iterate, so these two coefficients combine.
            d, e = self.prior_step, self.prior_penalty
            prior_spectral = d * self.spectral_residual(x, hr_msi)
            intermediate = (1 - d - d * e) * x + prior_spectral + d * (x - e * (noise + x))
            detail, _ = self.denoise(project(torch.cat((self.embed(intermediate), skip), dim=1)))
            proposal = detail + intermediate
            if observer is not None:
                observer(stage + 1, "prior", intermediate)
                observer(stage + 1, "proposal", proposal)
                observer(stage + 1, "prior_spectral_delta", prior_spectral)
                observer(stage + 1, "prior_noise_delta", -d * e * noise)
                observer(stage + 1, "refine_delta", detail)
            if stage == len(self.refine) - 1:
                return proposal

            residual = self.spatial_forward(x) - lr_hsi
            spatial = self.spatial_back(residual, output_size=x.shape[-2:])
            d, e = self.data_step, self.data_penalty
            data_spectral = self.spectral_residual(x, hr_msi)
            updated = (1 - d * e) * x + d * (data_spectral + spatial) + d * e * proposal
            if observer is not None:
                observer(stage + 1, "updated", updated)
                observer(stage + 1, "data_spatial_delta", d * spatial)
                observer(stage + 1, "data_spectral_delta", d * data_spectral)
                observer(stage + 1, "data_proposal_delta", d * e * (proposal - x))
            x = updated


def build_model(info):
    return MoGDCN(info["n_bands"], info["n_select_bands"], info["scale_ratio"])
