"""Torch-2.6 compatible implementation of PRFCoAM's custom four-scan Mamba.

This keeps the released PRFCoAM scan topology and parameterization, but replaces
its 2023 fused ``mamba_inner_fn_no_out_proj`` / causal-conv CUDA ABI with:

* PyTorch grouped causal conv1d for the short depthwise convolution; and
* the public ``selective_scan_fn`` from an installed, Torch-compatible
  ``mamba_ssm`` package for the SSM scan.

The goal is ABI compatibility only. PRFCoAM's v2 spectral two-direction scan,
v3 spatial four-direction scan, attention branches, SSM parameters and output
projections are preserved.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception as exc:  # pragma: no cover - environment-specific import error
    raise RuntimeError(
        "PRFCoAM compatibility mode needs a Torch-compatible mamba_ssm install "
        "that provides mamba_ssm.ops.selective_scan_interface.selective_scan_fn."
    ) from exc


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ChannelAttentionModule(nn.Module):
    def __init__(self, channel: int, reduction: int = 8):
        super().__init__()
        mid_channel = max(1, int(channel) // int(reduction))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.shared_MLP = nn.Sequential(
            nn.Linear(int(channel), mid_channel),
            nn.ReLU(),
            nn.Linear(mid_channel, int(channel)),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        avgout = self.shared_MLP(self.avg_pool(x).view(b, -1)).unsqueeze(2)
        maxout = self.shared_MLP(self.max_pool(x).view(b, -1)).unsqueeze(2)
        return self.sigmoid(avgout + maxout).reshape(b, 1, -1)


class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv2d = nn.Conv2d(2, 1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        return self.sigmoid(self.conv2d(torch.cat([avgout, maxout], dim=1)))


def _causal_depthwise_conv(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
) -> torch.Tensor:
    """Released Mamba causal-conv math using native PyTorch operators."""
    seqlen = x.shape[-1]
    width = weight.shape[-1]
    out = F.conv1d(
        x,
        weight,
        bias,
        padding=width - 1,
        groups=x.shape[1],
    )
    return F.silu(out[..., :seqlen])


def mamba_inner_fn_no_out_proj_compat(
    xz: torch.Tensor,
    conv1d_weight: torch.Tensor,
    conv1d_bias: torch.Tensor | None,
    x_proj_weight: torch.Tensor,
    delta_proj_weight: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor | None = None,
    C: torch.Tensor | None = None,
    D: torch.Tensor | None = None,
    *,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = True,
) -> torch.Tensor:
    """Equivalent unfused path for the author's mamba_inner_fn_no_out_proj.

    ``xz`` is B x (2*d_inner) x L. The first half is convolved and projected
    into dt/B/C, then scanned; the second half is the SiLU gate consumed by
    selective_scan_fn. No final output projection is applied here, matching the
    author's custom function.
    """
    if xz.ndim != 3:
        raise ValueError(f"expected BxCxL xz tensor, got {tuple(xz.shape)}")
    if xz.shape[1] % 2 != 0:
        raise ValueError(f"xz channel count must be even, got {xz.shape[1]}")

    x, z = xz.chunk(2, dim=1)
    if conv1d_weight.ndim != 3 or conv1d_weight.shape[1] != 1:
        raise ValueError(f"expected depthwise conv weight Dx1xW, got {tuple(conv1d_weight.shape)}")

    x = _causal_depthwise_conv(x, conv1d_weight, conv1d_bias)
    batch, d_inner, seqlen = x.shape
    delta_rank = delta_proj_weight.shape[1]
    d_state = A.shape[-1]

    x_dbl = F.linear(rearrange(x, "b d l -> (b l) d"), x_proj_weight)
    if B is None:
        B_flat = x_dbl[:, delta_rank : delta_rank + d_state]
        B = rearrange(B_flat, "(b l) n -> b n l", b=batch, l=seqlen).contiguous()
    if C is None:
        C_flat = x_dbl[:, -d_state:]
        C = rearrange(C_flat, "(b l) n -> b n l", b=batch, l=seqlen).contiguous()

    delta = delta_proj_weight @ x_dbl[:, :delta_rank].t()
    delta = rearrange(delta, "d (b l) -> b d l", b=batch, l=seqlen).contiguous()

    return selective_scan_fn(
        x,
        delta,
        A,
        B,
        C,
        D,
        z=z,
        delta_bias=delta_bias,
        delta_softplus=delta_softplus,
        return_last_state=False,
    )


class Mamba(nn.Module):
    """Released PRFCoAM custom Mamba with a modern selective-scan backend."""

    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=1,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,
        layer_idx=None,
        device=None,
        dtype=None,
        bimamba_type="v2",
        if_devide_out=False,
        init_layer_scale=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.Cin = 102
        self.Cout = 4
        self.d_model = int(d_model)
        self.d_state = int(d_state)
        self.d_conv = int(d_conv)
        self.expand = int(expand)
        self.d_inner = self.d_model
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else int(dt_rank)
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.bimamba_type = bimamba_type
        self.if_devide_out = if_devide_out
        self.init_layer_scale = init_layer_scale

        if init_layer_scale is not None:
            self.gamma = nn.Parameter(init_layer_scale * torch.ones((self.d_model,)), requires_grad=True)

        self.conv2d_1 = nn.Conv2d(
            self.d_model, self.d_model, kernel_size=3, padding=1, bias=conv_bias
        )
        self.conv1d = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.activation = "silu"
        self.act = nn.SiLU()
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(dt_init)

        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True

        def make_A_log():
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            out = nn.Parameter(torch.log(A))
            out._no_weight_decay = True
            return out

        self.A_log = make_A_log()
        self.A_b_log = make_A_log()
        self.A_c_log = make_A_log()
        self.A_d_log = make_A_log()

        def make_D():
            out = nn.Parameter(torch.ones(self.d_inner, device=device))
            out._no_weight_decay = True
            return out

        self.D = make_D()
        self.D_b = make_D()
        self.D_c = make_D()
        self.D_d = make_D()

        self.conv1d_b = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.conv1d_c = nn.Conv1d(
            self.d_inner,
            self.d_inner,
            kernel_size=self.d_conv,
            groups=self.d_inner,
            padding=self.d_conv - 1,
            bias=conv_bias,
            **factory_kwargs,
        )
        self.x_proj_b = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.x_proj_c = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj_b = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.dt_proj_c = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

        self.sa = SpatialAttentionModule().to(device)
        self.ca = ChannelAttentionModule(self.Cin).to(device)

    def _scan(self, xz, conv, x_proj, dt_proj, A_log, D):
        return mamba_inner_fn_no_out_proj_compat(
            xz,
            conv.weight,
            conv.bias,
            x_proj.weight,
            dt_proj.weight,
            -torch.exp(A_log.float()),
            None,
            None,
            D.float(),
            delta_bias=dt_proj.bias.float(),
            delta_softplus=True,
        )

    def forward(self, hidden_states, inference_params=None, extra_emb1=None, extra_emb2=None):
        del inference_params, extra_emb1, extra_emb2
        b, c, h, w = hidden_states.shape

        if self.bimamba_type == "v2":
            patch_size = 2
            if h % patch_size != 0 or w % patch_size != 0 or h != w:
                raise ValueError(f"PRFCoAM v2 expects even square feature maps, got {h}x{w}")
            xa = self.ca(hidden_states)
            N = h // patch_size
            x = hidden_states.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
            x = (
                x.contiguous()
                .view(b, c, N * N, patch_size, patch_size)
                .permute(0, 2, 3, 4, 1)
                .reshape(b * N * N, patch_size * patch_size, c)
            )
            xa_f = xa.repeat(1, h * w, 1).reshape(b * N * N, patch_size * patch_size, c)
            xz = torch.cat([x, xa_f], dim=1)

            out = self._scan(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
            out_b = self._scan(
                xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b
            )
            merged = out + out_b.flip([-1])
            if self.if_devide_out:
                merged = merged / 2
            out = F.linear(
                rearrange(merged, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias
            )
            out = (
                out.reshape(b, N, N, c, patch_size, patch_size)
                .permute(0, 3, 1, 4, 2, 5)
                .reshape(b, c, h, w)
            )

        elif self.bimamba_type == "v3":
            xa = self.sa(hidden_states)
            xz = torch.cat([hidden_states, xa.repeat(1, c, 1, 1)], dim=1).flatten(2, 3)
            out = self._scan(xz, self.conv1d, self.x_proj, self.dt_proj, self.A_log, self.D)
            out_b = self._scan(
                xz.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_b_log, self.D_b
            )

            bb, cc, ll = xz.shape
            side = round(math.sqrt(ll))
            if side * side != ll:
                raise ValueError(f"PRFCoAM v3 expects square feature maps, sequence length={ll}")
            xzb = xz.view(bb, cc, side, side).transpose(2, 3).flatten(2, 3)
            out_c = self._scan(
                xzb, self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_c_log, self.D_c
            )
            out_d = self._scan(
                xzb.flip([-1]), self.conv1d_b, self.x_proj_b, self.dt_proj_b, self.A_d_log, self.D_d
            )

            out_c = out_c.view(bb, cc // 2, side, side).transpose(2, 3).flatten(2, 3)
            out_d = (
                out_d.view(bb, cc // 2, side, side)
                .transpose(2, 3)
                .flatten(2, 3)
                .flip([-1])
            )
            merged = out + out_b.flip([-1]) + out_c + out_d
            if self.if_devide_out:
                merged = merged / 2
            out = F.linear(
                rearrange(merged, "b d l -> b l d"), self.out_proj.weight, self.out_proj.bias
            )
            out = out.permute(0, 2, 1).reshape(bb, self.Cout, side, side)
        else:
            raise ValueError(f"unsupported PRFCoAM bimamba_type: {self.bimamba_type}")

        if self.init_layer_scale is not None:
            out = out * self.gamma
        return out, xa


__all__ = [
    "ChannelAttentionModule",
    "SpatialAttentionModule",
    "Mamba",
    "mamba_inner_fn_no_out_proj_compat",
]
