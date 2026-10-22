"""PSRF-DiffNet reproduction core.

Clean reimplementation of the registration/fusion blocks released with:
"Progressive Synergistic Registration and Fusion Diffusion Network for
Unregistered Hyperspectral and Multispectral Image Fusion" (TGRS 2025).

The original repository hard-codes CUDA device, 102 HSI bands, 4 MSI bands and
160x160 patches. This version keeps the CRN -> FRN -> complementary fusion
topology but makes those dimensions configurable.
"""

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, force_projection: bool = False):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (
            nn.Identity()
            if in_ch == out_ch and not force_projection
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.body(x) + self.skip(x), inplace=True)


class CrossAffineAttention(nn.Module):
    """Official CRN cross-attention followed by a 2x3 affine localizer.

    The released model forms Q/K/V projections for both modalities, uses HSI
    queries and values with MSI keys, and regresses one affine matrix per patch.
    """

    def __init__(self, dim: int, token_count: int, num_heads: int = 4):
        super().__init__()
        if dim % num_heads:
            raise ValueError("dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv_hsi = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_msi = nn.Linear(dim, dim * 3, bias=False)
        self.localizer = nn.Sequential(
            nn.Linear(dim * token_count, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 6),
        )
        # Identity initialization is a numerical-stability adaptation; the
        # learned affine-transform mechanism remains the released CRN design.
        nn.init.zeros_(self.localizer[-1].weight)
        with torch.no_grad():
            self.localizer[-1].bias.copy_(
                torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
            )

    def forward(self, hsi: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        b, n, c = hsi.shape
        qkv_h = self.qkv_hsi(hsi).reshape(
            b, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        q_hsi, _k_hsi, v_hsi = qkv_h.unbind(0)
        qkv_m = self.qkv_msi(msi).reshape(
            b, n, 3, self.num_heads, self.head_dim
        ).permute(2, 0, 3, 1, 4)
        _q_msi, k_msi, _v_msi = qkv_m.unbind(0)

        attn = torch.matmul(q_hsi, k_msi.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        out = torch.matmul(attn, v_hsi).transpose(1, 2).reshape(b, n, c)
        return self.localizer(out.reshape(b, -1)).view(b, 2, 3)


class CoarseRegistrationNetwork(nn.Module):
    """CRN from PSRF-DiffNet: LR cross-modal attention -> affine warp."""

    def __init__(
        self,
        hr_patch_size: int,
        scale_ratio: int,
        hsi_channels: int,
        msi_channels: int,
        dim: int = 256,
        num_heads: int = 4,
    ):
        super().__init__()
        if hr_patch_size % scale_ratio:
            raise ValueError("hr_patch_size must be divisible by scale_ratio")
        lr_size = hr_patch_size // scale_ratio
        self.hsi_embed = nn.Conv2d(hsi_channels, dim, 3, 1, 1)
        self.msi_embed = nn.Conv2d(msi_channels, dim, 3, 1, 1)
        self.pos = nn.Parameter(torch.zeros(1, lr_size * lr_size, dim))
        # The official CRN shares one LayerNorm across the two modalities.
        self.norm = nn.LayerNorm(dim)
        self.attn = CrossAffineAttention(dim, lr_size * lr_size, num_heads)

    def forward(
        self,
        image_to_warp: torch.Tensor,
        lr_hsi: torch.Tensor,
        lr_msi: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.hsi_embed(lr_hsi).flatten(2).transpose(1, 2) + self.pos
        m = self.msi_embed(lr_msi).flatten(2).transpose(1, 2) + self.pos
        theta = self.attn(self.norm(h), self.norm(m))
        grid = F.affine_grid(theta, image_to_warp.shape, align_corners=False)
        warped = F.grid_sample(
            image_to_warp,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped, theta


def _neighborhood_stack(x: torch.Tensor) -> torch.Tensor:
    """Return 3x3 neighborhoods as Bx(9C)xHxW in upstream ordering."""
    if x.ndim != 4:
        raise ValueError("expected BxCxHxW")
    p = F.pad(x, (2, 2, 2, 2))
    offsets = (
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3),
    )
    h, w = x.shape[-2:]
    return torch.cat([p[:, :, oy:oy+h, ox:ox+w] for oy, ox in offsets], dim=1)


def build_registration_patches(
    lr_hsi_up: torch.Tensor,
    hr_msi: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build the HSI_Patch/MSI_Patch2 tensors used by the released FRN."""
    hsi_patch = _neighborhood_stack(lr_hsi_up)
    msi_patch = _neighborhood_stack(hr_msi)
    b, c9, h, w = msi_patch.shape
    p = F.pad(msi_patch, (2, 2, 2, 2))
    offsets = (
        (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
        (1, 3), (2, 3), (3, 3),
    )
    msi_patch2 = torch.stack(
        [p[:, :, oy:oy+h, ox:ox+w] for oy, ox in offsets],
        dim=-1,
    )
    msi_patch2 = msi_patch2.reshape(b, c9, h * w, 9)
    return hsi_patch, msi_patch2


class FineRegistrationAttention(nn.Module):
    """Official FRN cross-attention over the nine local MSI candidates."""

    def __init__(self, dim: int = 256):
        super().__init__()
        self.scale = dim ** -0.5
        self.qkv_hsi = nn.Linear(dim, dim * 3, bias=False)
        self.qkv_msi = nn.Linear(dim, dim * 3, bias=False)

    def forward(
        self,
        hsi_feat: torch.Tensor,
        msi_feat: torch.Tensor,
    ) -> torch.Tensor:
        b, c, h, w = hsi_feat.shape
        hsi_tokens = hsi_feat.flatten(2).transpose(1, 2)
        msi_candidates = msi_feat.permute(0, 2, 3, 1)
        # msi_candidates: B x (H*W) x 9 x C
        q = self.qkv_hsi(hsi_tokens).reshape(b, h * w, 3, c)[..., 0, :]
        k = self.qkv_msi(msi_candidates).reshape(b, h * w, 9, 3, c)[..., 1, :]
        score = (q.unsqueeze(-1) * k.permute(0, 1, 3, 2)).sum(dim=2) * self.scale
        index = score.softmax(dim=-1).argmax(dim=-1)
        # The released implementation swaps row/column offsets and feeds
        # pixel offsets directly to grid_sample. Keep the 3x3 hard matching,
        # but use explicit (dx,dy) ordering and a normalized sampling grid.
        dx = index.remainder(3).float() - 1.0
        dy = torch.div(index, 3, rounding_mode="floor").float() - 1.0
        return torch.stack((dx, dy), dim=-1).view(b, h, w, 2)


def _pixel_offset_grid(offset_px: torch.Tensor) -> torch.Tensor:
    """Convert local content offsets in pixels to a normalized sampling grid."""
    b, h, w, _ = offset_px.shape
    theta = torch.zeros(b, 2, 3, device=offset_px.device, dtype=offset_px.dtype)
    theta[:, 0, 0] = 1.0
    theta[:, 1, 1] = 1.0
    base = F.affine_grid(theta, (b, 1, h, w), align_corners=False)
    grid = base.clone()
    grid[..., 0] -= 2.0 * offset_px[..., 0] / float(w)
    grid[..., 1] -= 2.0 * offset_px[..., 1] / float(h)
    return grid


class FineRegistrationNetwork(nn.Module):
    """FRN local 3x3 matching followed by differentiable local warping."""

    def __init__(self, hsi_channels: int, msi_channels: int, dim: int = 256):
        super().__init__()
        self.hsi_embed = nn.Conv2d(hsi_channels * 9, dim, 1)
        self.msi_embed = nn.Conv2d(msi_channels * 9, dim, 1)
        self.reduce = nn.Conv2d(hsi_channels * 2, hsi_channels, 1)
        self.attn = FineRegistrationAttention(dim)

    def forward(
        self,
        x_pair: torch.Tensor,
        hsi_patch: torch.Tensor,
        msi_patch2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.reduce(x_pair)
        h = self.hsi_embed(hsi_patch)
        m = self.msi_embed(msi_patch2)
        offset = self.attn(h, m)
        grid = _pixel_offset_grid(offset)
        warped = F.grid_sample(
            x,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        return warped, offset


def _pca_repeat(x: torch.Tensor, channels: int = 256) -> torch.Tensor:
    b, c, h, w = x.shape
    tokens = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
    _, _, v = torch.pca_lowrank(tokens, q=1, center=True)
    y = torch.bmm(tokens, v[:, :, :1])
    return y.reshape(b, 1, h, w).repeat(1, channels, 1, 1)


class ConvGuidedFilter(nn.Module):
    def __init__(self, channels: int = 256):
        super().__init__()
        self.box = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        nn.init.constant_(self.box.weight, 1.0)
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
        )

    def forward(self, x: torch.Tensor, guide: torch.Tensor) -> torch.Tensor:
        y = _pca_repeat(guide, x.shape[1])
        n = self.box(torch.ones_like(x)).clamp_min(1e-6)
        mean_x = self.box(x) / n
        mean_y = self.box(y) / n
        cov_xy = self.box(x * y) / n - mean_x * mean_y
        var_x = self.box(x * x) / n - mean_x * mean_x
        a = self.conv_a(torch.cat([cov_xy, var_x], dim=1))
        b = mean_y - a * mean_x
        return a * x + b


class StructureTensor(nn.Module):
    def __init__(self, msi_channels: int):
        super().__init__()
        self.pre = nn.Conv2d(msi_channels, msi_channels, 3, 1, 1)
        sx = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32
        )
        sy = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32
        )
        self.register_buffer("sx", sx.view(1, 1, 3, 3).repeat(msi_channels, 1, 1, 1))
        self.register_buffer("sy", sy.view(1, 1, 3, 3).repeat(msi_channels, 1, 1, 1))
        self.norm = nn.BatchNorm2d(1)
        self.channels = msi_channels

    def forward(self, y: torch.Tensor) -> torch.Tensor:
        y = F.pad(self.pre(y), (2, 2, 2, 2), mode="replicate")
        gx = F.conv2d(y, self.sx, padding=1, groups=self.channels).sum(1, keepdim=True)
        gy = F.conv2d(y, self.sy, padding=1, groups=self.channels).sum(1, keepdim=True)
        a = gx.square() + 0.1
        b = gx * gy + 0.1
        d = gy.square() + 0.1
        attn = a * d - b.square()
        attn = attn[:, :, 2:-2, 2:-2]
        return self.norm(attn)


class ComplementaryFusion(nn.Module):
    """Spatial structure + spectral/channel + guided-filter fusion."""

    def __init__(self, patch_size: int, hsi_channels: int, msi_channels: int):
        super().__init__()
        self.patch_size = patch_size
        self.structure = StructureTensor(msi_channels)
        self.msi_1 = ResidualBlock(msi_channels, 64)
        self.msi_2 = ResidualBlock(128, 64)

        self.hsi_in = nn.Conv2d(hsi_channels, 256, 3, 1, 1)
        # Official ResBlock_HSI_1 uses same_shape=False, hence a 1x1 skip.
        self.hsi_1 = ResidualBlock(256, 256, force_projection=True)
        self.channel_attn = nn.Conv1d(patch_size * patch_size, 256, 3, padding=1)
        self.hsi_merge = ResidualBlock(512, 256)
        self.guided = ConvGuidedFilter(256)

        self.out = nn.Sequential(
            ResidualBlock(256 + 256 + 64, 256),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 256),
            nn.ReLU(inplace=True),
            ResidualBlock(256, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, hsi_channels, 3, 1, 1),
            nn.Conv2d(hsi_channels, hsi_channels, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor, msi: torch.Tensor) -> torch.Tensor:
        spa = self.structure(msi)
        m = self.msi_1(msi)
        m = self.msi_2(torch.cat([m, m * spa], dim=1))

        h = self.hsi_1(self.hsi_in(x))
        b, c, hh, ww = h.shape
        if hh != self.patch_size or ww != self.patch_size:
            raise ValueError(
                f"PSRF fusion was built for {self.patch_size}x{self.patch_size}, "
                f"got {hh}x{ww}"
            )
        tokens = h.flatten(2).transpose(1, 2)
        spec = self.channel_attn(tokens)
        h_att = torch.matmul(tokens, spec).transpose(1, 2).reshape(b, c, hh, ww)
        h = self.hsi_merge(torch.cat([h, h_att], dim=1))
        guided = h + self.guided(h, m)
        return self.out(torch.cat([h, guided, m], dim=1))


class CCFNet(nn.Module):
    """Fine registration + complementary spatial/spectral fusion."""

    def __init__(self, patch_size: int, hsi_channels: int, msi_channels: int):
        super().__init__()
        self.frn = FineRegistrationNetwork(hsi_channels, msi_channels)
        self.fuse = ComplementaryFusion(patch_size, hsi_channels, msi_channels)

    def forward(
        self,
        hsi_pair: torch.Tensor,
        hr_msi: torch.Tensor,
        hsi_patch: torch.Tensor,
        msi_patch2: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x, offset = self.frn(hsi_pair, hsi_patch, msi_patch2)
        return self.fuse(x, hr_msi), offset


__all__ = [
    "CCFNet",
    "CoarseRegistrationNetwork",
    "build_registration_patches",
]
