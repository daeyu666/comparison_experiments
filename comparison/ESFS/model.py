"""ESFS spatial/frequency fusion, independently implemented after Xu et al.
(TGRS 2025) and checked against the active path in the released CAVE model.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

ARCHITECTURE_ID = "esfs-cave-spatial-frequency-v1"
UPSTREAM_COMMIT = "38bd5b373438d2d89cadeb3ae58f4a67f6fffbda"


def conv(i, o, k=3, *, groups=1, stride=1, padding=None):
    return nn.Conv2d(i, o, k, stride=stride, padding=k // 2 if padding is None else padding, groups=groups)


def image_norm(x, norm):
    return norm(x.movedim(1, -1)).movedim(-1, 1)


def split_windows(x, size):
    b, c, h, w = x.shape
    return (x.reshape(b, c, h // size, size, w // size, size)
            .permute(0, 2, 4, 3, 5, 1).reshape(-1, size * size, c))


def merge_windows(tokens, size, h, w):
    c = tokens.shape[-1]
    return (tokens.reshape(-1, h // size, w // size, size, size, c)
            .permute(0, 5, 1, 3, 2, 4).reshape(-1, c, h, w))


class CondensedAttention(nn.Module):
    def __init__(self, width, window=8, heads=6):
        super().__init__()
        self.window, self.heads, self.head_width = window, heads, width // heads
        self.reduce = conv(width, width, 2, stride=2, groups=width, padding=0)
        self.local = conv(width, width, groups=width)
        self.mix = conv(width, width, 1)
        self.norm = nn.LayerNorm(width)
        self.q = nn.Linear(width, width)
        self.kv = nn.Linear(width, 2 * width)
        self.out = conv(width, width, 1)
        small = window // 4
        self.relative = nn.Parameter(torch.empty((2 * small - 1) ** 2, heads))
        nn.init.trunc_normal_(self.relative, std=0.02)
        qr, qc = torch.meshgrid(torch.arange(window) // 4, torch.arange(window) // 4, indexing="ij")
        kr, kc = torch.meshgrid(torch.arange(small), torch.arange(small), indexing="ij")
        idx = ((qr.flatten()[:, None] - kr.flatten() + small - 1) * (2 * small - 1)
               + qc.flatten()[:, None] - kc.flatten() + small - 1)
        self.register_buffer("relative_index", idx, persistent=False)

    def forward(self, tokens, mask):
        n, q_len, width = tokens.shape
        z = tokens.transpose(1, 2).reshape(n, width, self.window, self.window).contiguous()
        z = self.reduce(self.reduce(z))
        z = self.mix(self.local(z)).flatten(2).transpose(1, 2)
        z = F.gelu(self.norm(z))
        q = self.q(tokens).reshape(n, q_len, self.heads, self.head_width).transpose(1, 2)
        k, v = self.kv(z).reshape(n, -1, 2, self.heads, self.head_width).permute(2, 0, 3, 1, 4)
        logits = (q * self.head_width ** -0.5) @ k.transpose(-2, -1)
        logits = logits + self.relative[self.relative_index].permute(2, 0, 1)
        if mask is not None:
            windows = mask.shape[0]
            logits = (logits.unflatten(0, (-1, windows)) + mask[None, :, None]).flatten(0, 1)
        z = (logits.softmax(-1) @ v).transpose(1, 2)
        # Released CSAM packs the attention output directly as C x H x W.
        z = z.reshape(n, width, self.window, self.window)
        return self.out(z).flatten(2).transpose(1, 2)


def shifted_mask(x, window, shift):
    h, w = x.shape[-2:]
    # Region IDs after cyclic window partition, at full and condensed sizes.
    def regions(side, block, offset):
        axis = torch.arange(side, device=x.device)
        return (axis >= side - block).long() + (axis >= side - offset).long()

    def labels(height, width, block, offset):
        rows = regions(height, block, offset)
        cols = regions(width, block, offset)
        return split_windows((3 * rows[:, None] + cols)[None, None], block).squeeze(-1)

    q = labels(h, w, window, shift)
    k = labels(h // 4, w // 4, window // 4, shift // 4)
    return (q[:, :, None] != k[:, None, :]).to(x.dtype) * -100


def stochastic_residual(x, probability, training):
    if not training or probability == 0:
        return x
    keep = 1 - probability
    mask = x.new_empty((x.shape[0], 1, 1, 1)).bernoulli_(keep)
    return x * (mask / keep)


def grouped_spatial(width):
    group = width // 4
    return nn.Sequential(conv(width, group, groups=group), nn.LeakyReLU(0.2),
                         conv(group, width, groups=group))


class SpatialFusion(nn.Module):
    def __init__(self, width, shift, drop):
        super().__init__()
        self.window, self.shift, self.drop = 8, shift, drop
        self.pre = nn.LayerNorm(width)
        self.attn = CondensedAttention(width, self.window)
        self.local = grouped_spatial(width)
        self.post = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.fuse = conv(2 * width, width, 1)

    def forward(self, x, mask):
        shift = self.shift
        normalized = image_norm(x, self.pre)
        shifted = normalized.roll((-shift, -shift), (-2, -1)) if shift else normalized
        windows = self.attn(split_windows(shifted, self.window), mask if shift else None)
        spatial = merge_windows(windows, self.window, *x.shape[-2:])
        if shift:
            spatial = spatial.roll((shift, shift), (-2, -1))
        shortcut = x + stochastic_residual(spatial, self.drop, self.training)
        ff = self.mlp(self.post(shortcut.movedim(1, -1))).movedim(-1, 1)
        shortcut = shortcut + stochastic_residual(ff, self.drop, self.training)
        return self.fuse(torch.cat((shortcut, self.local(normalized)), dim=1))


def pack_complex(z):
    return torch.stack((z.real, z.imag), dim=2).flatten(1, 2)


def unpack_complex(x):
    return torch.complex(x[:, 0::2], x[:, 1::2])


class FrequencyFusion(nn.Module):
    def __init__(self, width, drop):
        super().__init__()
        reduced = width // 2
        self.drop = drop
        self.norm = nn.LayerNorm(width)
        self.project_in = nn.Sequential(conv(width, reduced, 1), nn.LeakyReLU(0.2))
        self.project_out = conv(reduced, width, 1)
        self.spectral_gate = nn.Sequential(conv(width, reduced, 1), nn.ReLU(), conv(reduced, width, 1), nn.Sigmoid())
        self.amplitude = nn.Sequential(conv(reduced, reduced, 1), nn.LeakyReLU(0.2))
        self.phase = nn.Sequential(conv(reduced, reduced), nn.LeakyReLU(0.2))
        self.mix = nn.Sequential(conv(width, width, 1), nn.LeakyReLU(0.2))
        self.local = grouped_spatial(width)
        self.fuse = conv(2 * width, width, 1)

    def forward(self, x):
        z = self.project_in(image_norm(x, self.norm))
        spectrum = torch.fft.rfft2(z, norm="ortho")
        amplitude, phase = spectrum.abs(), spectrum.angle()
        pooled = torch.polar(amplitude.amax(dim=(-2, -1), keepdim=True),
                             phase.mean(dim=(-2, -1), keepdim=True))
        gate = unpack_complex(self.spectral_gate(pack_complex(pooled)))
        adjusted = torch.polar(amplitude + self.amplitude(amplitude), phase + self.phase(phase)) * gate
        refined = unpack_complex(self.mix(pack_complex(adjusted)))
        z = self.project_out(z + torch.fft.irfft2(refined, s=z.shape[-2:], norm="ortho"))
        z = x + stochastic_residual(z, self.drop, self.training)
        return self.fuse(torch.cat((z, self.local(x)), dim=1))


class SpatialFrequencyGroup(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.in_norm = nn.LayerNorm(width)
        self.out_norm = nn.LayerNorm(width)
        self.group_out = conv(width, width)
        self.body_out = conv(width, width)
        # The release constructs the first of its configured [6, 6, 6]
        # layers, then reuses that same six-block body at all three stages.
        drop = torch.linspace(0, 0.1, 18).tolist()
        self.spatial = nn.ModuleList(SpatialFusion(width, 4 * (i % 2), drop[i]) for i in range(6))
        self.frequency = nn.ModuleList(FrequencyFusion(width, drop[i]) for i in range(6))

    def forward(self, x, mask):
        stem = image_norm(x, self.in_norm)
        z = stem
        for spatial, frequency in zip(self.spatial, self.frequency):
            z = z + spatial(z, mask)
            z = z + frequency(z)
        z = stem + self.group_out(z)
        return x + self.body_out(image_norm(z, self.out_norm))


class ESFS(nn.Module):
    def __init__(self, hsi_channels: int, msi_channels: int, scale: int = 4):
        super().__init__()
        width = 180
        self.scale = scale
        self.hsi_channels, self.msi_channels = hsi_channels, msi_channels
        self.hsi_head = conv(hsi_channels, width)
        self.msi_head = nn.Sequential(conv(msi_channels, 64), nn.ReLU(), conv(64, width))
        self.join = conv(2 * width, width)
        self.body = SpatialFrequencyGroup(width)
        self.aggregate = nn.ModuleList(conv(width * i, width) for i in (2, 3, 4))
        self.output = conv(width, hsi_channels)

    def forward(self, lr_hsi, hr_msi):
        h, w = hr_msi.shape[-2:]
        if h % 8 or w % 8:
            raise ValueError("ESFS requires HR dimensions divisible by its window size (8)")
        low = F.interpolate(lr_hsi, scale_factor=self.scale, mode="bicubic", align_corners=False)
        stem = self.hsi_head(low)
        features = self.join(torch.cat((stem, self.msi_head(hr_msi)), dim=1))
        mask = shifted_mask(features, 8, 4)
        dense = [stem]
        for project in self.aggregate:
            features = self.body(features, mask)
            features = project(torch.cat((*dense, features), dim=1))
            dense.append(features)
        return self.output(stem + features)


def build_model(info):
    return ESFS(info["n_bands"], info["n_select_bands"], info["scale_ratio"])
