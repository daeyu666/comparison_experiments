"""Self-contained PRFCoAM four-scan Mamba compatibility backend.

The released PRFCoAM code was written against Mamba-1.0.1-era fused CUDA
extensions. Those binary extensions are ABI-incompatible with the repository's
Torch-2.6 + cu124 environment on the target machine.  This module therefore
preserves the released PRFCoAM v2/v3 scan topology and Mamba parameterization,
but implements both operations needed by the custom block with native PyTorch:

* depthwise causal conv1d via ``torch.nn.functional.conv1d``;
* selective state-space scan via an equivalent parallel affine-prefix scan.

No installed ``mamba_ssm``, ``causal_conv1d_cuda`` or ``selective_scan_cuda``
package is required.  ``base/`` remains untouched.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat


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


def _affine_prefix_scan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Parallel inclusive scan for ``x_t = a_t * x_(t-1) + b_t``.

    ``a`` and ``b`` are B x D x L x N.  An affine transition is represented by
    the pair (a, b).  Composition is associative:

        (a2, b2) o (a1, b1) = (a2*a1, b2 + a2*b1)

    so a Hillis-Steele prefix scan evaluates every recurrent state in
    O(log L) tensor rounds rather than a Python loop over all L positions.
    With zero initial state, the prefix transform's ``b`` component is exactly
    the selective-scan state at each position.
    """
    if a.shape != b.shape or a.ndim != 4:
        raise ValueError(f"affine scan expects equal BxDxLxN tensors, got {a.shape} and {b.shape}")

    length = a.shape[2]
    offset = 1
    while offset < length:
        a_left = a[:, :, :-offset, :]
        b_left = b[:, :, :-offset, :]
        a_right = a[:, :, offset:, :]
        b_right = b[:, :, offset:, :]

        composed_a = a_right * a_left
        composed_b = b_right + a_right * b_left

        a = torch.cat((a[:, :, :offset, :], composed_a), dim=2)
        b = torch.cat((b[:, :, :offset, :], composed_b), dim=2)
        offset <<= 1
    return b


def selective_scan_torch(
    u: torch.Tensor,
    delta: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    D: torch.Tensor | None = None,
    z: torch.Tensor | None = None,
    delta_bias: torch.Tensor | None = None,
    delta_softplus: bool = False,
    return_last_state: bool = False,
):
    """Native-PyTorch equivalent of Mamba's real-valued ``selective_scan_ref``.

    PRFCoAM uses real A and input-dependent B/C tensors of shape B x N x L.
    The constant and grouped B/C layouts supported by the public Mamba
    reference function are retained as well.
    """
    if u.ndim != 3 or delta.shape != u.shape:
        raise ValueError(f"u/delta must both be BxDxL, got {u.shape} and {delta.shape}")
    if A.ndim != 2 or A.shape[0] != u.shape[1]:
        raise ValueError(f"A must be DxN with D={u.shape[1]}, got {A.shape}")
    if A.is_complex() or B.is_complex() or C.is_complex():
        raise NotImplementedError("PRFCoAM compatibility backend only needs real-valued selective scan")

    dtype_in = u.dtype
    u_f = u.float()
    delta_f = delta.float()
    A_f = A.float()

    if delta_bias is not None:
        delta_f = delta_f + delta_bias.float().unsqueeze(-1)
    if delta_softplus:
        delta_f = F.softplus(delta_f)

    batch, dim, _ = u_f.shape

    B_f = B.float()
    C_f = C.float()
    if B_f.ndim == 4:
        if dim % B_f.shape[1] != 0:
            raise ValueError(f"B groups {B_f.shape[1]} do not divide model dim {dim}")
        B_f = repeat(B_f, "b g n l -> b (g h) n l", h=dim // B_f.shape[1])
    if C_f.ndim == 4:
        if dim % C_f.shape[1] != 0:
            raise ValueError(f"C groups {C_f.shape[1]} do not divide model dim {dim}")
        C_f = repeat(C_f, "b g n l -> b (g h) n l", h=dim // C_f.shape[1])

    # Discretized transition and input terms used by the official reference:
    #   state_t = exp(delta_t * A) * state_(t-1) + delta_t * B_t * u_t
    deltaA = torch.exp(torch.einsum("bdl,dn->bdln", delta_f, A_f))
    if B_f.ndim == 2:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta_f, B_f, u_f)
    elif B_f.ndim == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta_f, B_f, u_f)
    elif B_f.ndim == 4:
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta_f, B_f, u_f)
    else:
        raise ValueError(f"unsupported B layout {B.shape}")

    states = _affine_prefix_scan(deltaA, deltaB_u)

    if C_f.ndim == 2:
        y = torch.einsum("bdln,dn->bdl", states, C_f)
    elif C_f.ndim == 3:
        y = torch.einsum("bdln,bnl->bdl", states, C_f)
    elif C_f.ndim == 4:
        y = torch.einsum("bdln,bdnl->bdl", states, C_f)
    else:
        raise ValueError(f"unsupported C layout {C.shape}")

    out = y
    if D is not None:
        out = out + u_f * D.float().view(1, -1, 1)
    if z is not None:
        out = out * F.silu(z.float())
    out = out.to(dtype=dtype_in)

    if return_last_state:
        return out, states[:, :, -1, :]
    return out


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
    """Unfused equivalent of the author's ``mamba_inner_fn_no_out_proj``."""
    if xz.ndim != 3:
        raise ValueError(f"expected BxCxL xz tensor, got {tuple(xz.shape)}")
    if xz.shape[1] % 2 != 0:
        raise ValueError(f"xz channel count must be even, got {xz.shape[1]}")

    x, z = xz.chunk(2, dim=1)
    if conv1d_weight.ndim != 3 or conv1d_weight.shape[1] != 1:
        raise ValueError(f"expected depthwise conv weight Dx1xW, got {tuple(conv1d_weight.shape)}")

    x = _causal_depthwise_conv(x, conv1d_weight, conv1d_bias)
    batch, _, seqlen = x.shape
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

    return selective_scan_torch(
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
    """Released PRFCoAM custom Mamba with a self-contained PyTorch backend."""

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
    "selective_scan_torch",
]
