"""UAFL CFDA / DCNv2 implementation for the current Torch environment.

This file preserves the released UAFL CFDA topology and equations.  The only
backend substitution is mmcv.ops.modulated_deform_conv2d ->
torchvision.ops.deform_conv2d(mask=...), because the paper/released mmcv-full
1.6.1 binary is not compatible with Torch 2.6 + CUDA 12.4.

No learnable layer, offset/mask parameterization, flow predictor, positional
encoding, deformable-group setting, or warp convention is changed.
"""
from __future__ import annotations

import math
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

try:
    from torchvision.ops import deform_conv2d as _tv_deform_conv2d
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "UAFL requires torchvision.ops.deform_conv2d with modulated-mask support. "
        "Use the torchvision build matching the installed PyTorch."
    ) from exc


def modulated_deform_conv2d(
    input: torch.Tensor,
    offset: torch.Tensor,
    mask: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    stride,
    padding,
    dilation,
    groups: int,
    deformable_groups: int,
) -> torch.Tensor:
    """mmcv-call-compatible wrapper around torchvision's modulated DCNv2 op."""
    if int(groups) != 1:
        raise NotImplementedError("Released UAFL uses convolution groups=1 only")
    kh, kw = weight.shape[-2:]
    expected_offset = 2 * int(deformable_groups) * kh * kw
    expected_mask = int(deformable_groups) * kh * kw
    if offset.shape[1] != expected_offset:
        raise ValueError(f"offset channels={offset.shape[1]}, expected {expected_offset}")
    if mask.shape[1] != expected_mask:
        raise ValueError(f"mask channels={mask.shape[1]}, expected {expected_mask}")
    if input.shape[1] % int(deformable_groups) != 0:
        raise ValueError(
            f"input channels {input.shape[1]} must be divisible by deformable_groups="
            f"{deformable_groups}; the paper configuration uses a divisible feature width"
        )
    return _tv_deform_conv2d(
        input,
        offset,
        weight,
        bias=bias,
        stride=stride,
        padding=padding,
        dilation=dilation,
        mask=mask,
    )


class DCNv2(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation=1,
        deformable_groups=1,
    ):
        super().__init__()
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.deformable_groups = int(deformable_groups)
        self.weight = nn.Parameter(
            torch.empty(self.out_channels, self.in_channels, *self.kernel_size)
        )
        self.bias = nn.Parameter(torch.empty(self.out_channels))
        self.reset_parameters()

    def reset_parameters(self):
        n = self.in_channels
        for k in self.kernel_size:
            n *= k
        stdv = 1.0 / math.sqrt(n)
        self.weight.data.uniform_(-stdv, stdv)
        self.bias.data.zero_()

    def forward(self, input, offset, mask):
        return modulated_deform_conv2d(
            input,
            offset,
            mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            1,
            self.deformable_groups,
        )


def generate_pe(decimal_offset, num_freqs=8, temperature=10000.0):
    """Released UAFL sine/cosine sub-pixel positional encoding."""
    decimal_offset = decimal_offset.permute(0, 2, 3, 1)
    device, dtype = decimal_offset.device, decimal_offset.dtype
    freq_bands = temperature ** (
        torch.arange(num_freqs, device=device, dtype=dtype) / num_freqs
    )
    freq_bands = freq_bands.view(1, 1, 1, 1, -1)
    inputs = decimal_offset.unsqueeze(-1) * freq_bands
    embedded = torch.cat([torch.sin(inputs), torch.cos(inputs)], dim=-1)
    return embedded.flatten(-2).permute(0, 3, 1, 2)


def flow_warp(x, flow, interp_mode="bilinear", padding_mode="zeros", align_corners=True):
    """Released UAFL backward feature warp; flow is Bx2xHxW in pixel units."""
    flow = flow.permute(0, 2, 3, 1)
    if x.size()[-2:] != flow.size()[1:3]:
        raise ValueError(f"warp size mismatch: x={x.shape}, flow={flow.shape}")
    _, _, h, w = x.size()
    grid_y, grid_x = torch.meshgrid(
        torch.arange(0, h, device=x.device, dtype=x.dtype),
        torch.arange(0, w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    grid = torch.stack((grid_x, grid_y), 2)
    vgrid = grid.unsqueeze(0) + flow
    vgrid_x = 2.0 * vgrid[..., 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[..., 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    return F.grid_sample(
        x,
        vgrid_scaled,
        mode=interp_mode,
        padding_mode=padding_mode,
        align_corners=align_corners,
    )


class PyramidFlowPredictor(nn.Module):
    """Released two-level coarse pyramid flow/similarity predictor (CPFP)."""
    def __init__(self, in_channels):
        super().__init__()
        self.num_flow_channels = 2
        self.num_sim_channels = 1
        self.level1_predictor = nn.Conv2d(
            in_channels * 2, self.num_flow_channels, 3, 1, 1
        )
        self.level0_predictor = nn.Conv2d(
            in_channels * 2,
            self.num_flow_channels + self.num_sim_channels,
            3,
            1,
            1,
        )
        self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)
        self.init_weights()

    def init_weights(self):
        self.level1_predictor.weight.data.zero_()
        self.level1_predictor.bias.data.zero_()
        self.level0_predictor.weight.data.zero_()
        self.level0_predictor.bias.data.zero_()

    def forward(self, ref_feat, target_feat):
        ref_l1 = self.downsample(ref_feat)
        target_l1 = self.downsample(target_feat)
        flow_l1 = self.level1_predictor(torch.cat([ref_l1, target_l1], 1)) * 2.0
        flow_l0 = F.interpolate(
            flow_l1, scale_factor=2, mode="bilinear", align_corners=False
        )
        warped_ref = flow_warp(ref_feat, flow_l0)
        residual_out = self.level0_predictor(torch.cat([warped_ref, target_feat], 1))
        residual_flow = residual_out[:, :2]
        pre_sim = torch.sigmoid(residual_out[:, 2:])
        return flow_l0 + residual_flow, pre_sim


class DCN_Refine_With_Prior_flow(DCNv2):
    """Released UAFL coarse-to-fine deformable aggregation (CFDA)."""
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=1,
        dilation=1,
        deformable_groups=1,
        max_residue_magnitude=10,
    ):
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            deformable_groups,
        )
        self.max_residue_magnitude = max_residue_magnitude
        self.prior_predictor = PyramidFlowPredictor(in_channels)
        num_pe_channels = 2 * 2 * 8
        refine_in_channels = in_channels + in_channels + num_pe_channels
        dcn_offset_channels = self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        dcn_mask_channels = self.deformable_groups * self.kernel_size[0] * self.kernel_size[1]
        refine_out_channels = dcn_offset_channels + dcn_mask_channels
        self.refinement_net = nn.Sequential(
            nn.Conv2d(refine_in_channels, in_channels, 3, 1, 1),
            nn.LeakyReLU(0.1, inplace=True),
            nn.Conv2d(in_channels, refine_out_channels, 3, 1, 1),
        )
        self.init_weights()
        self.last_prior_flow = None
        self.last_prior_similarity = None

    def init_weights(self):
        self.refinement_net[-1].weight.data.zero_()
        self.refinement_net[-1].bias.data.zero_()

    def forward(self, ref_feat, target_feat):
        pre_flow, pre_sim = self.prior_predictor(ref_feat, target_feat)
        self.last_prior_flow = pre_flow.detach()
        self.last_prior_similarity = pre_sim.detach()

        offset_decimal = pre_flow - torch.floor(pre_flow)
        pe_decimal = generate_pe(offset_decimal)
        warped_ref_feat = flow_warp(ref_feat, pre_flow)
        refinement_input = torch.cat(
            [target_feat, warped_ref_feat, pe_decimal], dim=1
        )
        refinement_out = self.refinement_net(refinement_input)

        dcn_offset_channels = (
            self.deformable_groups * 2 * self.kernel_size[0] * self.kernel_size[1]
        )
        residual_offset = refinement_out[:, :dcn_offset_channels]
        residual_mask_logits = refinement_out[:, dcn_offset_channels:]
        if self.max_residue_magnitude:
            residual_offset = self.max_residue_magnitude * torch.tanh(residual_offset)

        k_sq = self.kernel_size[0] * self.kernel_size[1]
        expanded = pre_flow.unsqueeze(1).repeat(1, k_sq, 1, 1, 1)
        b, _, _, h, w = expanded.size()
        prior_offset = expanded.view(b, -1, h, w).repeat(
            1, self.deformable_groups, 1, 1
        )
        final_offset = prior_offset + residual_offset
        pre_sim_repeated = pre_sim.repeat(
            1, self.deformable_groups * k_sq, 1, 1
        )
        final_mask = torch.sigmoid(residual_mask_logits * pre_sim_repeated)

        return modulated_deform_conv2d(
            ref_feat,
            final_offset,
            final_mask,
            self.weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            1,
            self.deformable_groups,
        )


__all__ = [
    "DCNv2",
    "DCN_Refine_With_Prior_flow",
    "PyramidFlowPredictor",
    "flow_warp",
    "generate_pe",
    "modulated_deform_conv2d",
]
