"""Independent PyTorch PSRT (TGRS 2023) W2W model implementation.

Architecture reference: https://github.com/shangqideng/PSRT
Original model/model_SR.py blob: 07c41e4290719edc2fc1fd15ef56fec69d662d42.
Matches 2 pyramid groups, 3 W2W blocks/group, 8/4/2 windows, 32 latent
channels, 8 heads, the released q-k scale, and residual HSI prediction.
This is an independently written architectural reimplementation rather than
a verbatim copy of the upstream source.
"""
import torch
from torch import nn


def _tile(x, size):
    batch, height, width, channels = x.shape
    if height % size or width % size:
        raise ValueError("PSRT windows require dimensions divisible by window size")
    return (x.reshape(batch, height // size, size, width // size, size, channels)
            .permute(0, 1, 3, 2, 4, 5).reshape(-1, size * size, channels))


def _untile(tokens, size, height, width):
    count = (height // size) * (width // size)
    batch = tokens.shape[0] // count
    channels = tokens.shape[-1]
    return (tokens.reshape(batch, height // size, width // size, size, size, channels)
            .permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels))


def _window_reindex(x, window_size, inverse=False):
    """Rearrange nonoverlapping half-window tiles in PSRT's W2W ordering."""
    batch, channels, height, width = x.shape
    half = window_size // 2
    if (window_size < 2 or window_size % 2 or
            height % window_size or width % window_size):
        raise ValueError("Invalid PSRT shuffle geometry")
    tile_h, tile_w = height // half, width // half
    patches = (_tile(x.permute(0, 2, 3, 1), half)
               .reshape(batch, tile_h, tile_w, half * half * channels))
    result = torch.empty_like(patches)
    if inverse:
        result[:, 0::2, 0::2] = patches[:, :tile_h // 2, :tile_w // 2]
        result[:, 0::2, 1::2] = patches[:, :tile_h // 2, tile_w // 2:]
        result[:, 1::2, 0::2] = patches[:, tile_h // 2:, :tile_w // 2]
        result[:, 1::2, 1::2] = patches[:, tile_h // 2:, tile_w // 2:]
    else:
        result[:, :tile_h // 2, :tile_w // 2] = patches[:, 0::2, 0::2]
        result[:, :tile_h // 2, tile_w // 2:] = patches[:, 0::2, 1::2]
        result[:, tile_h // 2:, :tile_w // 2] = patches[:, 1::2, 0::2]
        result[:, tile_h // 2:, tile_w // 2:] = patches[:, 1::2, 1::2]
    pixels = result.reshape(batch * tile_h * tile_w, half * half, channels)
    return _untile(pixels, half, height, width).permute(0, 3, 1, 2)


def Win_Shuffle(x, win_size):
    return _window_reindex(x, win_size, inverse=False)


def Win_Reshuffle(x, win_size):
    return _window_reindex(x, win_size, inverse=True)


class WindowAttention(nn.Module):
    def __init__(self, channels=32, heads=8, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.heads = heads
        self.norm1 = nn.LayerNorm(channels)
        self.qkv = nn.Linear(channels, channels * 3)
        self.proj = nn.Linear(channels, channels)
        self.norm2 = nn.LayerNorm(channels)
        self.ffn = nn.Sequential(
            nn.Linear(channels, channels * 4),
            nn.GELU(),
            nn.Linear(channels * 4, channels)
        )
        # Official PSRT's Window_Attention defaults to qk_scale=4.
        self.q_scale = 4.0

    def forward(self, x):
        batch, channels, height, width = x.shape
        spatial = x.permute(0, 2, 3, 1).reshape(batch, height * width, channels)
        windows = _tile(self.norm1(spatial).reshape(batch, height, width, channels),
                        self.window_size)
        group_count, tokens_per_window, _ = windows.shape
        packed = self.qkv(windows).reshape(
            group_count, tokens_per_window, 3, self.heads, channels // self.heads
        )
        queries, keys, values = packed.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (queries * self.q_scale) @ keys.transpose(-2, -1)
        weights = torch.softmax(scores, dim=-1)
        attended = (weights @ values).transpose(1, 2).reshape(
            group_count, tokens_per_window, channels
        )
        attended = self.proj(attended)
        spatial = spatial + _untile(
            attended, self.window_size, height, width
        ).reshape(batch, height * width, channels)
        spatial = spatial + self.ffn(self.norm2(spatial))
        return spatial.reshape(batch, height, width, channels).permute(0, 3, 1, 2)


class W2WBlock(nn.Module):
    def __init__(self, channels=32, heads=8, window_size=8):
        super().__init__()
        self.window_size = window_size
        self.attn1 = WindowAttention(channels, heads, window_size)
        self.attn2 = WindowAttention(channels, heads, window_size)
        self.attn3 = WindowAttention(channels, heads, window_size)

    def forward(self, x):
        skip = x
        x = self.attn1(x)
        x = Win_Shuffle(x, self.window_size)
        x = self.attn2(x)
        x = Win_Reshuffle(x, self.window_size)
        return self.attn3(x) + skip


class PyramidGroup(nn.Module):
    def __init__(self, channels=32, heads=8):
        super().__init__()
        self.layers = nn.ModuleList([
            W2WBlock(channels, heads, window_size=window)
            for window in (8, 4, 2)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class PSRTnet(nn.Module):
    """Two pyramids of 3 W2W blocks; predict an HSI residual from HR inputs."""
    def __init__(self, args):
        super().__init__()
        self.img_size = int(getattr(args, "patch_size", 64))
        self.in_channels = int(getattr(args, "n_bands", 31))
        self.msi_channels = int(getattr(args, "n_bands_rgb", 3))
        self.embed = 32
        if self.img_size < 8 or self.img_size % 8:
            raise ValueError("PSRT patch size must be divisible by 8")
        if self.in_channels < 1 or self.msi_channels < 1:
            raise ValueError("PSRT HSI/MSI channel counts must be positive")
        self.input_conv = nn.Conv2d(
            self.in_channels + self.msi_channels, 32, 3, padding=1
        )
        self.groups = nn.Sequential(PyramidGroup(), PyramidGroup())
        self.final_conv = nn.Sequential(
            nn.Conv2d(32, self.in_channels, 3, padding=1),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        # Corresponds to the released PSRT weight initialization.
        for layer in self.groups.modules():
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0.0, 0.01)
            elif isinstance(layer, nn.Conv2d) and layer.bias is not None:
                nn.init.zeros_(layer.bias)
        if self.input_conv.bias is not None:
            nn.init.zeros_(self.input_conv.bias)
        if self.final_conv[0].bias is not None:
            nn.init.zeros_(self.final_conv[0].bias)

    def forward(self, rgb, lms):
        if rgb.shape[-2:] != lms.shape[-2:]:
            raise ValueError("PSRT needs MSI and upsampled HSI on the same HR grid")
        features = self.input_conv(torch.cat((lms, rgb), dim=1))
        features = self.groups(features)
        return self.final_conv(features) + lms

    def name(self):
        return "PSRT"
