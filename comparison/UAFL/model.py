"""Paper-faithful UAFL network adapted only at the reference input channels.

Reference:
  Y. Zhang et al., "Enhancing Unregistered Hyperspectral Image Super-Resolution
  via Unmixing-based Abundance Fusion Learning," CVPR 2026.

The released topology is retained: SVD unmixing, one down/up multi-scale
encoder-decoder, CFDA, repeated SCACA, SCMF, abundance residual mapping, and
mixing with the fixed SVD endmembers.  The paper-sized configuration used here
is dim=128, stage=1, num_blocks=(2,1), K=3.  This is consistent with Fig. 3,
the released construction pattern, deformable_groups=8, and the paper's ~5.94M
parameter report.  The public class default dim=28 is not compatible with its
own fixed deformable_groups=8 and is treated as a release/debug default rather
than the paper run configuration.

Only ``ref_channels`` is made dynamic: 3 for the original RGB paper setting,
4 for PaviaU/IKONOS, and 8 for WV2 experiments.
"""
from __future__ import annotations

import math
import warnings

import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange

from dcn_compat import DCN_Refine_With_Prior_flow

PAPER_DIM = 128
PAPER_STAGE = 1
PAPER_NUM_BLOCKS = (2, 1)
PAPER_NUM_ENDMEMBERS = 3
PAPER_DEFORMABLE_GROUPS = 8
PAPER_PARAMETER_M = 5.94


def _to_2tuple(x):
    return x if isinstance(x, tuple) else (x, x)


def mix(A_hat, E_y):
    bs, r, h, w = A_hat.shape
    _, _, c = E_y.shape
    A_hat_f_m = A_hat.reshape(bs, r, h * w)
    E_y_reshaped = E_y.permute(0, 2, 1)
    return torch.bmm(E_y_reshaped, A_hat_f_m).reshape(bs, c, h, w)


def Unmix_svd_3d(y, Rr=3):
    """Released UAFL SVD unmixing, with the modern equivalent SVD API."""
    b, c, h, w = y.shape
    y_flat = y.reshape(b, c, -1)
    # torch.svd(..., some=True) used by the release is mathematically the thin
    # SVD. torch.linalg.svd(full_matrices=False) is its maintained equivalent.
    U, _S, _Vh = torch.linalg.svd(y_flat, full_matrices=False)
    E = U[:, :, :Rr].permute(0, 2, 1)
    A = (E @ y_flat).reshape(b, Rr, h, w)
    return A, E


class GELU(nn.Module):
    def forward(self, x):
        return F.gelu(x)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, *args, **kwargs):
        return self.fn(self.norm(x), *args, **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, mult=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim * mult, 1, 1, bias=False),
            GELU(),
            nn.Conv2d(
                dim * mult,
                dim * mult,
                3,
                1,
                1,
                bias=False,
                groups=dim * mult,
            ),
            GELU(),
            nn.Conv2d(dim * mult, dim, 1, 1, bias=False),
        )

    def forward(self, x):
        return self.net(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    if H % window_size or W % window_size:
        raise ValueError(
            f"UAFL window attention requires H/W divisible by {window_size}; got {H}x{W}"
        )
    x = x.view(
        B,
        H // window_size,
        window_size,
        W // window_size,
        window_size,
        C,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(-1, window_size, window_size, C)
    )


def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(
        B,
        H // window_size,
        W // window_size,
        window_size,
        window_size,
        -1,
    )
    return (
        x.permute(0, 1, 3, 2, 4, 5)
        .contiguous()
        .view(B, H, W, -1)
    )


class CrossWindowAttention(nn.Module):
    """Released SACA window cross-attention: reference modulates V."""
    def __init__(
        self,
        dim,
        window_size,
        num_heads,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros(
                (2 * window_size[0] - 1) * (2 * window_size[1] - 1),
                num_heads,
            )
        )
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid(coords_h, coords_w, indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += self.window_size[0] - 1
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        self.register_buffer(
            "relative_position_index", relative_coords.sum(-1), persistent=True
        )
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, ref, mask=None):
        B_, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B_, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(
            self.window_size[0] * self.window_size[1],
            self.window_size[0] * self.window_size[1],
            -1,
        )
        attn = attn + relative_position_bias.permute(2, 0, 1).unsqueeze(0)
        # The released implementation accepts ``mask`` but does not apply it.
        attn = self.attn_drop(self.softmax(attn))
        ref_reshaped = ref.reshape(
            B_, N, self.num_heads, C // self.num_heads
        ).permute(0, 2, 1, 3)
        v_modulated = v * ref_reshaped
        x = (attn @ v_modulated).transpose(1, 2).reshape(B_, N, C)
        return self.proj_drop(self.proj(x))


class CrossSwinTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        input_resolution,
        num_heads,
        window_size=7,
        shift_size=0,
        qkv_bias=True,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.dim = dim
        self.input_resolution = (
            input_resolution
            if isinstance(input_resolution, tuple)
            else (input_resolution, input_resolution)
        )
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError("shift_size must be in [0, window_size)")
        self.norm1 = norm_layer(dim)
        self.attn = CrossWindowAttention(
            dim,
            window_size=_to_2tuple(self.window_size),
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # Keep the released attention-mask buffer construction. The released
        # CrossWindowAttention does not consume this mask, so runtime H/W stay dynamic.
        if self.shift_size > 0:
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (
                slice(0, -self.window_size),
                slice(-self.window_size, -self.shift_size),
                slice(-self.shift_size, None),
            )
            w_slices = h_slices
            cnt = 0
            for hs in h_slices:
                for ws in w_slices:
                    img_mask[:, hs, ws, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size).view(
                -1, self.window_size * self.window_size
            )
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(
                attn_mask != 0, float(-100.0)
            ).masked_fill(attn_mask == 0, 0.0)
        else:
            attn_mask = None
        self.register_buffer("attn_mask", attn_mask, persistent=True)

    def forward(self, x, ref):
        B, H, W, C = x.shape
        x = self.norm1(x.reshape(B, H * W, C)).reshape(B, H, W, C)
        ref = self.norm1(ref.reshape(B, H * W, C)).reshape(B, H, W, C)
        if self.shift_size > 0:
            shifted_x = torch.roll(
                x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
            shifted_ref = torch.roll(
                ref, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)
            )
        else:
            shifted_x, shifted_ref = x, ref
        x_windows = window_partition(shifted_x, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )
        ref_windows = window_partition(shifted_ref, self.window_size).view(
            -1, self.window_size * self.window_size, C
        )
        attn_windows = self.attn(x_windows, ref_windows, mask=self.attn_mask)
        attn_windows = attn_windows.view(
            -1, self.window_size, self.window_size, C
        )
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift_size > 0:
            return torch.roll(
                shifted_x,
                shifts=(self.shift_size, self.shift_size),
                dims=(1, 2),
            )
        return shifted_x


class CrossHSA(nn.Module):
    """Released hierarchical SACA: W-MCA + shifted W-MCA + FFN."""
    def __init__(self, dim, stage=1):
        super().__init__()
        self.wa = CrossSwinTransformerBlock(
            dim=dim,
            input_resolution=256 // (2 ** stage),
            num_heads=2 ** stage,
            window_size=8,
            shift_size=0,
        )
        self.swa = CrossSwinTransformerBlock(
            dim=dim,
            input_resolution=256 // (2 ** stage),
            num_heads=2 ** stage,
            window_size=8,
            shift_size=4,
        )
        self.pn = PreNorm(dim, FeedForward(dim=dim))

    def forward(self, x, ref):
        x = self.wa(x, ref) + x
        x = self.swa(x, ref) + x
        x = self.pn(x) + x
        return x.permute(0, 3, 1, 2)


class MA(nn.Module):
    def __init__(self, n_feat):
        super().__init__()
        self.depth_conv = nn.Conv2d(
            n_feat, n_feat, kernel_size=5, padding=2, bias=True, groups=n_feat
        )

    def forward(self, mask_3d):
        attn_map = torch.sigmoid(self.depth_conv(mask_3d))
        return mask_3d * attn_map + mask_3d


class CrossSSM_AB(nn.Module):
    """Released SCACA block: spatial abundance + channel abundance attention."""
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        attention_type="full",
    ):
        super().__init__()
        self.num_heads = heads
        self.dim_head = dim_head
        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_k = nn.Linear(dim, dim_head * heads, bias=False)
        self.to_v = nn.Linear(dim, dim_head * heads, bias=False)
        self.rescale = nn.Parameter(torch.ones(heads, 1, 1))
        self.proj = nn.Linear(dim_head * heads, dim, bias=True)
        self.pos_emb = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
            GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim),
        )
        self.ma = MA(dim)
        self.sa = CrossHSA(dim)
        self.sa_conv = nn.Conv2d(
            dim, dim, 5, 1, 2, groups=dim, bias=False
        )
        self.dim = dim
        self.attention_type = attention_type

    def forward(self, x_in, spa_g=None, ref=None):
        b, h, w, c = x_in.shape
        ref_attn = self.ma(ref.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        if b != 0:
            ref_attn = ref_attn[0].expand([b, h, w, c])
        if self.attention_type == "base":
            x = x_in.reshape(b, h * w, c)
        else:
            x_mid_out = self.sa(x_in, ref_attn)
            x_sa_emb = self.sa_conv(spa_g) + spa_g
            if x_sa_emb.shape[3] != x_mid_out.shape[3]:
                raise RuntimeError(
                    "Released shift_back branch was for coded-aperture inputs and is "
                    "not expected in reference HSI SR"
                )
            x = (x_mid_out * x_sa_emb).permute(0, 2, 3, 1).reshape(
                b, h * w, c
            )
        q_inp = self.to_q(x)
        k_inp = self.to_k(x)
        v_inp = self.to_v(x)
        q, k, v, ref_attn_h = map(
            lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.num_heads),
            (q_inp, k_inp, v_inp, ref_attn.flatten(1, 2)),
        )
        if self.attention_type == "full":
            v = v * ref_attn_h
        q = F.normalize(q.transpose(-2, -1), dim=-1, p=2)
        k = F.normalize(k.transpose(-2, -1), dim=-1, p=2)
        v = v.transpose(-2, -1)
        attn = (k @ q.transpose(-2, -1)) * self.rescale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).permute(0, 3, 1, 2).reshape(
            b, h * w, self.num_heads * self.dim_head
        )
        out_c = self.proj(x).view(b, h, w, c)
        out_p = self.pos_emb(
            v_inp.reshape(b, h, w, c).permute(0, 3, 1, 2)
        ).permute(0, 2, 3, 1)
        return out_c + out_p


class CrossCAB(nn.Module):
    def __init__(
        self,
        dim,
        dim_head=64,
        heads=8,
        num_blocks=1,
        attention_type="full",
    ):
        super().__init__()
        self.blocks = nn.ModuleList([])
        for _ in range(num_blocks):
            self.blocks.append(
                nn.ModuleList(
                    [
                        CrossSSM_AB(
                            dim=dim,
                            dim_head=dim_head,
                            heads=heads,
                            attention_type=attention_type,
                        ),
                        PreNorm(dim, FeedForward(dim=dim)),
                    ]
                )
            )

    def forward(self, x, ref):
        x = x.permute(0, 2, 3, 1)
        for attn, ff in self.blocks:
            x = attn(x, ref, ref.permute(0, 2, 3, 1)) + x
            x = ff(x) + x
        return x.permute(0, 3, 1, 2)


class ModulatedSpatialSpectralFusion(nn.Module):
    """Released SCMF with parallel spatial and channel modulation."""
    def __init__(self, c_in):
        super().__init__()
        self.value_spatial = nn.Sequential(
            nn.Conv2d(
                c_in * 2,
                c_in * 2,
                3,
                padding=1,
                groups=c_in * 2,
                bias=False,
            ),
            nn.LeakyReLU(),
            nn.Conv2d(c_in * 2, c_in, 1, bias=False),
            nn.LeakyReLU(),
        )
        self.value_spectral = nn.Sequential(
            nn.Conv2d(c_in * 2, c_in, 1, bias=False)
        )
        self.spatial_attention_head = nn.Sequential(
            nn.Conv2d(c_in * 2, 1, 3, padding=1, bias=False), nn.Sigmoid()
        )
        self.spectral_attention_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c_in * 2, c_in, 1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, fea, fea_en):
        _B, _C, H, W = fea.shape
        if fea_en.shape[2:] != (H, W):
            fea_en = F.interpolate(
                fea_en, size=(H, W), mode="bilinear", align_corners=False
            )
        x = torch.cat([fea, fea_en], dim=1)
        value_s = self.value_spatial(x)
        value_p = self.value_spectral(x)
        w_spatial = self.spatial_attention_head(x)
        w_spectral = self.spectral_attention_head(x)
        return value_s * w_spatial + value_p * w_spectral + fea


class UAFL(nn.Module):
    def __init__(
        self,
        dim=PAPER_DIM,
        stage=PAPER_STAGE,
        num_blocks=PAPER_NUM_BLOCKS,
        attention_type="full",
        numend=PAPER_NUM_ENDMEMBERS,
        ref_channels=3,
    ):
        super().__init__()
        self.dim = int(dim)
        self.stage = int(stage)
        self.numend = int(numend)
        self.ref_channels = int(ref_channels)
        num_blocks = tuple(int(x) for x in num_blocks)
        if len(num_blocks) != self.stage + 1:
            raise ValueError("num_blocks must have stage+1 entries")
        if self.dim % PAPER_DEFORMABLE_GROUPS:
            raise ValueError(
                f"dim={self.dim} must be divisible by deformable_groups="
                f"{PAPER_DEFORMABLE_GROUPS}"
            )

        self.embedding = nn.Conv2d(
            self.numend, self.dim, 3, 1, 1, bias=False
        )
        self.embedding2 = nn.Conv2d(
            self.ref_channels, self.dim, 3, 1, 1, bias=False
        )

        self.encoder_layers = nn.ModuleList([])
        dim_stage = self.dim
        for i in range(self.stage):
            self.encoder_layers.append(
                nn.ModuleList(
                    [
                        CrossCAB(
                            dim=dim_stage,
                            num_blocks=num_blocks[i],
                            dim_head=self.dim,
                            heads=dim_stage // self.dim,
                            attention_type=attention_type,
                        ),
                        nn.Conv2d(
                            dim_stage, dim_stage * 2, 4, 2, 1, bias=False
                        ),
                        nn.Conv2d(
                            dim_stage, dim_stage * 2, 4, 2, 1, bias=False
                        ),
                        DCN_Refine_With_Prior_flow(
                            dim_stage,
                            dim_stage,
                            kernel_size=(3, 3),
                            stride=1,
                            padding=1,
                            deformable_groups=PAPER_DEFORMABLE_GROUPS,
                        ),
                    ]
                )
            )
            dim_stage *= 2

        self.bottleneck = CrossCAB(
            dim=dim_stage,
            dim_head=self.dim,
            heads=dim_stage // self.dim,
            num_blocks=num_blocks[-1],
            attention_type=attention_type,
        )

        self.decoder_layers = nn.ModuleList([])
        for i in range(self.stage):
            self.decoder_layers.append(
                nn.ModuleList(
                    [
                        nn.ConvTranspose2d(
                            dim_stage,
                            dim_stage // 2,
                            stride=2,
                            kernel_size=2,
                            padding=0,
                            output_padding=0,
                        ),
                        ModulatedSpatialSpectralFusion(dim_stage // 2),
                        CrossCAB(
                            dim=dim_stage // 2,
                            num_blocks=num_blocks[self.stage - 1 - i],
                            dim_head=self.dim,
                            heads=(dim_stage // 2) // self.dim,
                            attention_type=attention_type,
                        ),
                    ]
                )
            )
            dim_stage //= 2

        self.mapping = nn.Conv2d(
            self.dim, self.numend, 3, 1, 1, bias=False
        )
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

    def forward(self, x, ref=None):
        if ref is None:
            raise ValueError("UAFL requires an HR reference image")
        if x.shape[-2:] != ref.shape[-2:]:
            raise ValueError(
                f"UAFL expects upsampled HSI and HR reference on one grid; "
                f"got {x.shape[-2:]} vs {ref.shape[-2:]}"
            )
        A_hat, E_y = Unmix_svd_3d(x, self.numend)
        fea = self.lrelu(self.embedding(A_hat))
        ref_fea = self.lrelu(self.embedding2(ref))

        fea_encoder = []
        refs = []
        for CrossCAB_, FeaDownSample, RefDownSample, RefConv in self.encoder_layers:
            ref_fea = RefConv(ref_fea, fea)
            fea = CrossCAB_(fea, ref_fea)
            refs.append(ref_fea)
            fea_encoder.append(fea)
            fea = FeaDownSample(fea)
            ref_fea = RefDownSample(ref_fea)

        fea = self.bottleneck(fea, ref_fea)
        for i, (FeaUpSample, Fusion, CrossCAB_) in enumerate(self.decoder_layers):
            fea = FeaUpSample(fea)
            fea = Fusion(fea, fea_encoder[self.stage - 1 - i])
            ref_fea = refs[self.stage - 1 - i]
            fea = CrossCAB_(fea, ref_fea)

        return mix(self.mapping(fea), E_y) + x


def build_uafl(ref_channels: int, *, paper_config: bool = True) -> UAFL:
    if not paper_config:
        warnings.warn(
            "Non-paper UAFL configurations are intentionally unsupported in the "
            "comparison launcher; instantiate UAFL directly for architecture studies."
        )
    return UAFL(
        dim=PAPER_DIM,
        stage=PAPER_STAGE,
        num_blocks=PAPER_NUM_BLOCKS,
        attention_type="full",
        numend=PAPER_NUM_ENDMEMBERS,
        ref_channels=int(ref_channels),
    )


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


__all__ = [
    "UAFL",
    "build_uafl",
    "parameter_count",
    "mix",
    "Unmix_svd_3d",
    "PAPER_DIM",
    "PAPER_STAGE",
    "PAPER_NUM_BLOCKS",
    "PAPER_NUM_ENDMEMBERS",
    "PAPER_DEFORMABLE_GROUPS",
    "PAPER_PARAMETER_M",
]
