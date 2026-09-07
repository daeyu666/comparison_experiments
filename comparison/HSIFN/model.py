"""HSIFN / HSI-RefSR reproduction adapted to the shared HSI-MSI benchmark.

Paper:
    Hyperspectral Image Super Resolution With Real Unaligned RGB Guidance
    IEEE TNNLS, 2025 (early access 2023).

This implementation keeps the key published design:
  * reference-to-HSI coarse flow alignment;
  * a second flow estimator for multi-scale feature alignment;
  * confidence masks on warped reference features;
  * QRNN3D HSI encoder/decoder fusion.

The original code assumes RGB guidance (3 channels). Here the reference encoder
and both FlowNets are generalized to arbitrary MSI channels (4 for IKONOS,
8 for WV2). The pseudo-reference produced from the upsampled HSI uses the exact
SRF matrix from comparison_experiments, so the alignment direction remains
HR-MSI -> HSI/GT coordinates.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Reference-image alignment blocks (architecture-faithful FlowNet path)
# -----------------------------------------------------------------------------


def _conv_act(
    in_ch: int,
    out_ch: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    activation: str = "leaky_relu",
) -> nn.Sequential:
    layers: List[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding)
    ]
    if activation == "leaky_relu":
        layers.append(nn.LeakyReLU(0.1, inplace=False))
    elif activation == "selu":
        layers.append(nn.SELU(inplace=False))
    elif activation == "relu":
        layers.append(nn.ReLU(inplace=False))
    elif activation != "linear":
        raise ValueError(f"unsupported activation: {activation}")
    return nn.Sequential(*layers)


def _flow_head(in_ch: int) -> nn.Conv2d:
    return nn.Conv2d(in_ch, 2, 3, 1, 1)


def _upsample(in_ch: int, out_ch: int) -> nn.ConvTranspose2d:
    return nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True)


def _leaky_deconv(in_ch: int, out_ch: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=True),
        nn.LeakyReLU(0.1, inplace=False),
    )


class FlowNet(nn.Module):
    """FlowNet used by the official HSIFN implementation.

    Both inputs have ``ref_channels`` channels and share the same spatial size.
    Returned flow fields match full, 1/2, 1/4 and 1/8 resolution reference
    feature maps.
    """

    def __init__(self, ref_channels: int):
        super().__init__()
        c = int(ref_channels)
        self.ref_channels = c

        self.conv1 = _conv_act(2 * c, 64, 7, 2, 3)
        self.conv2 = _conv_act(64, 128, 5, 2, 2)
        self.conv3 = _conv_act(128, 256, 5, 2, 2)
        self.conv3_1 = _conv_act(256, 256)
        self.conv4 = _conv_act(256, 512, 3, 2, 1)
        self.conv4_1 = _conv_act(512, 512)
        self.conv5 = _conv_act(512, 512, 3, 2, 1)
        self.conv5_1 = _conv_act(512, 512)
        self.conv6 = _conv_act(512, 1024, 3, 2, 1)
        self.conv6_1 = _conv_act(1024, 1024)

        self.flow6 = _flow_head(1024)
        self.flow6_up = _upsample(2, 2)
        self.deconv5 = _leaky_deconv(1024, 256)

        self.flow5 = _flow_head(770)
        self.flow5_up = _upsample(2, 2)
        self.deconv4 = _leaky_deconv(770, 256)

        self.flow4 = _flow_head(770)
        self.flow4_up = _upsample(2, 2)
        self.deconv3 = _leaky_deconv(770, 128)

        self.flow3 = _flow_head(386)
        self.flow3_up = _upsample(2, 2)
        self.deconv2 = _leaky_deconv(386, 64)

        self.flow2 = _flow_head(194)
        self.flow2_up = _upsample(2, 2)
        self.deconv1 = _leaky_deconv(194, 64)

        self.flow1 = _flow_head(130)
        self.flow1_up = _upsample(2, 2)
        self.deconv0 = _leaky_deconv(130, 64)

        # Official code hard-codes 69 = RGB(3) + 64 + 2.
        self.concat0_conv1 = _conv_act(c + 66, 16, 7, 1, 3, activation="selu")
        self.concat0_conv2 = _conv_act(16, 16, 7, 1, 3, activation="selu")
        self.flow_12 = _flow_head(16)

    @staticmethod
    def _cat_exact(parts: Sequence[torch.Tensor]) -> torch.Tensor:
        """Crop decoder tensors by at most one pixel when odd shapes occur."""
        h = min(x.shape[-2] for x in parts)
        w = min(x.shape[-1] for x in parts)
        return torch.cat([x[..., :h, :w] for x in parts], dim=1)

    def forward(self, target_like: torch.Tensor, reference: torch.Tensor) -> Dict[str, torch.Tensor]:
        if target_like.shape != reference.shape:
            raise ValueError(
                f"FlowNet inputs must match, got {tuple(target_like.shape)} and "
                f"{tuple(reference.shape)}"
            )
        x = torch.cat((target_like, reference), dim=1)
        conv1 = self.conv1(x)
        conv2 = self.conv2(conv1)
        conv3_1 = self.conv3_1(self.conv3(conv2))
        conv4_1 = self.conv4_1(self.conv4(conv3_1))
        conv5_1 = self.conv5_1(self.conv5(conv4_1))
        conv6_1 = self.conv6_1(self.conv6(conv5_1))

        flow6 = self.flow6(conv6_1)
        flow6_up = self.flow6_up(flow6)
        deconv5 = self.deconv5(conv6_1)

        concat5 = self._cat_exact((conv5_1, deconv5, flow6_up))
        flow5 = self.flow5(concat5)
        flow5_up = self.flow5_up(flow5)
        deconv4 = self.deconv4(concat5)

        concat4 = self._cat_exact((conv4_1, deconv4, flow5_up))
        flow4 = self.flow4(concat4)
        flow4_up = self.flow4_up(flow4)
        deconv3 = self.deconv3(concat4)

        concat3 = self._cat_exact((conv3_1, deconv3, flow4_up))
        flow3 = self.flow3(concat3)
        flow3_up = self.flow3_up(flow3)
        deconv2 = self.deconv2(concat3)

        concat2 = self._cat_exact((conv2, deconv2, flow3_up))
        flow2 = self.flow2(concat2)
        flow2_up = self.flow2_up(flow2)
        deconv1 = self.deconv1(concat2)

        concat1 = self._cat_exact((conv1, deconv1, flow2_up))
        flow1 = self.flow1(concat1)
        flow1_up = self.flow1_up(flow1)
        deconv0 = self.deconv0(concat1)

        concat0 = self._cat_exact((reference, deconv0, flow1_up))
        flow_12 = self.flow_12(self.concat0_conv2(self.concat0_conv1(concat0)))

        return {
            "flow_12_1": flow_12,
            "flow_12_2": flow1,
            "flow_12_3": flow2,
            "flow_12_4": flow3,
        }


def flow_warp(x: torch.Tensor, flow_px: torch.Tensor) -> torch.Tensor:
    """Backward-warp ``x`` with a dense pixel displacement field.

    The convention follows the HSIFN usage: the predicted field is defined on
    the target (HSI) grid and samples from the reference image.
    """
    if x.ndim != 4 or flow_px.ndim != 4 or flow_px.shape[1] != 2:
        raise ValueError("expected x=BxCxHxW and flow=Bx2xHxW")
    if flow_px.shape[-2:] != x.shape[-2:]:
        flow_px = F.interpolate(
            flow_px, size=x.shape[-2:], mode="bilinear", align_corners=True
        )

    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.arange(h, device=x.device, dtype=x.dtype),
        torch.arange(w, device=x.device, dtype=x.dtype),
        indexing="ij",
    )
    base = torch.stack((xx, yy), dim=-1).unsqueeze(0).expand(b, h, w, 2)
    coords = base + flow_px.permute(0, 2, 3, 1)
    if w > 1:
        gx = 2.0 * coords[..., 0] / float(w - 1) - 1.0
    else:
        gx = torch.zeros_like(coords[..., 0])
    if h > 1:
        gy = 2.0 * coords[..., 1] / float(h - 1) - 1.0
    else:
        gy = torch.zeros_like(coords[..., 1])
    grid = torch.stack((gx, gy), dim=-1)
    return F.grid_sample(
        x,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )


class ReferenceEncoder(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        c = int(in_channels)
        self.layer_f = _conv_act(c, 64, 5, 1, 2, activation="selu")
        self.conv1 = _conv_act(64, 64, 5, 1, 2, activation="selu")
        self.conv2 = _conv_act(64, 64, 5, 2, 2, activation="selu")
        self.conv3 = _conv_act(64, 64, 5, 2, 2, activation="selu")
        self.conv4 = _conv_act(64, 64, 5, 2, 2, activation="selu")

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        x = self.layer_f(x)
        c1 = self.conv1(x)
        c2 = self.conv2(c1)
        c3 = self.conv3(c2)
        c4 = self.conv4(c3)
        return c1, c2, c3, c4


# -----------------------------------------------------------------------------
# Confidence-mask blocks
# -----------------------------------------------------------------------------


def _mask_conv(in_ch: int, out_ch: int, k: int = 3, p: int = 1, activation: bool = True):
    layers: List[nn.Module] = [nn.Conv2d(in_ch, out_ch, k, 1, p)]
    if activation:
        layers.append(nn.ReLU(inplace=False))
    return nn.Sequential(*layers)


class MaskResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = _mask_conv(channels, channels)
        self.conv2 = _mask_conv(channels, channels, activation=False)
        self.act = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv2(self.conv1(x)) + x)


class MaskPredictor(nn.Module):
    def __init__(self, input_dim: int = 64, project_dim: int = 32, offset_dim: int = 32):
        super().__init__()
        self.project = _mask_conv(input_dim, project_dim, 1, 0)
        self.offset = nn.Sequential(
            _mask_conv(2, offset_dim),
            MaskResBlock(offset_dim),
        )
        self.weight = nn.Sequential(
            _mask_conv(project_dim * 2 + offset_dim, project_dim * 2),
            MaskResBlock(project_dim * 2),
            _mask_conv(project_dim * 2, 1, activation=False),
        )

    def forward(self, target_feat: torch.Tensor, ref_feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        if flow.shape[-2:] != target_feat.shape[-2:]:
            flow = F.interpolate(flow, size=target_feat.shape[-2:], mode="bilinear", align_corners=True)
        x = self.project(target_feat)
        r = self.project(ref_feat)
        f = self.offset(torch.remainder(flow, 1.0))
        return torch.sigmoid(self.weight(torch.cat((x, r, f), dim=1)))


# -----------------------------------------------------------------------------
# QRNN3D spectral-spatial HSI encoder/decoder (ported from HSI-RefSR)
# -----------------------------------------------------------------------------


class BasicConv3d(nn.Sequential):
    def __init__(self, in_channels: int, channels: int, k=3, s=1, p=1, bias=False, bn=False):
        super().__init__()
        if bn:
            self.add_module("bn", nn.BatchNorm3d(in_channels))
        self.add_module("conv", nn.Conv3d(in_channels, channels, k, s, p, bias=bias))


class BasicDeConv3d(nn.Sequential):
    def __init__(self, in_channels: int, channels: int, k=3, s=1, p=1, bias=False, bn=False):
        super().__init__()
        if bn:
            self.add_module("bn", nn.BatchNorm3d(in_channels))
        self.add_module("deconv", nn.ConvTranspose3d(in_channels, channels, k, s, p, bias=bias))


class UpsampleConv3d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, k=3, s=1, p=1, bias=False):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, k, s, p, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=(1, 2, 2), mode="trilinear", align_corners=True)
        return self.conv(x)


class QRNN3DLayer(nn.Module):
    def __init__(self, hidden_channels: int, conv_layer: nn.Module, act: str = "tanh"):
        super().__init__()
        self.hidden_channels = int(hidden_channels)
        self.conv = conv_layer
        self.act = act

    def _activate(self, z: torch.Tensor) -> torch.Tensor:
        if self.act == "tanh":
            return z.tanh()
        if self.act == "relu":
            return z.relu()
        if self.act == "none":
            return z
        raise ValueError(self.act)

    @staticmethod
    def _rnn_step(z: torch.Tensor, f: torch.Tensor, h: Optional[torch.Tensor]) -> torch.Tensor:
        return (1.0 - f) * z if h is None else f * h + (1.0 - f) * z

    def forward(self, inputs: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        gates = self.conv(inputs)
        z, f = gates.split(self.hidden_channels, dim=1)
        z = self._activate(z)
        f = f.sigmoid()
        zs = z.split(1, 2)
        fs = f.split(1, 2)
        h = None
        outputs: List[torch.Tensor] = []
        if reverse:
            for zz, ff in zip(reversed(zs), reversed(fs)):
                h = self._rnn_step(zz, ff, h)
                outputs.insert(0, h)
        else:
            for zz, ff in zip(zs, fs):
                h = self._rnn_step(zz, ff, h)
                outputs.append(h)
        return torch.cat(outputs, dim=2)


class BiQRNN3DLayer(QRNN3DLayer):
    def forward(self, inputs: torch.Tensor, reverse: bool = False) -> torch.Tensor:
        del reverse
        gates = self.conv(inputs)
        z, f1, f2 = gates.split(self.hidden_channels, dim=1)
        z = self._activate(z)
        f1, f2 = f1.sigmoid(), f2.sigmoid()
        zs = z.split(1, 2)

        left: List[torch.Tensor] = []
        h = None
        for zz, ff in zip(zs, f1.split(1, 2)):
            h = self._rnn_step(zz, ff, h)
            left.append(h)

        right: List[torch.Tensor] = []
        h = None
        for zz, ff in zip(reversed(zs), reversed(f2.split(1, 2))):
            h = self._rnn_step(zz, ff, h)
            right.insert(0, h)
        return torch.cat(left, 2) + torch.cat(right, 2)


class BiQRNNConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels: int, hidden_channels: int, k=3, s=1, p=1, bn=False, act="tanh"):
        super().__init__(
            hidden_channels,
            BasicConv3d(in_channels, hidden_channels * 3, k, s, p, bn=bn),
            act=act,
        )


class BiQRNNDeConv3D(BiQRNN3DLayer):
    def __init__(self, in_channels: int, hidden_channels: int, k=3, s=1, p=1, bias=True, bn=False, act="tanh"):
        super().__init__(
            hidden_channels,
            BasicDeConv3d(in_channels, hidden_channels * 3, k, s, p, bias=bias, bn=bn),
            act=act,
        )


class QRNNConv3D(QRNN3DLayer):
    def __init__(self, in_channels: int, hidden_channels: int, k=3, s=1, p=1, bn=False, act="tanh"):
        super().__init__(
            hidden_channels,
            BasicConv3d(in_channels, hidden_channels * 2, k, s, p, bn=bn),
            act=act,
        )


class QRNNUpsampleConv3d(QRNN3DLayer):
    def __init__(self, in_channels: int, hidden_channels: int, k=3, s=1, p=1, bn=False, act="tanh"):
        super().__init__(
            hidden_channels,
            UpsampleConv3d(in_channels, hidden_channels * 2, k, s, p, bias=False),
            act=act,
        )


class HSIEncoder(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.feat_extractor = BiQRNNConv3D(1, channels, bn=False, act="tanh")
        self.layers = nn.ModuleList()
        c = channels
        for _ in range(3):
            self.layers.append(QRNNConv3D(c, 2 * c, k=3, s=(1, 2, 2), p=1, bn=False, act="tanh"))
            c *= 2

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], bool]:
        xs: List[torch.Tensor] = []
        x = self.feat_extractor(x)
        xs.append(x)
        reverse = False
        for layer in self.layers:
            x = layer(x, reverse=reverse)
            reverse = not reverse
            xs.append(x)
        return xs, reverse


class HSIDecoder(nn.Module):
    """HSIFN decoder equivalent to official HSIDecoder4."""

    def __init__(self):
        super().__init__()
        self.up1 = QRNNUpsampleConv3d(128 + 64, 64, bn=False, act="tanh")
        self.up2 = QRNNUpsampleConv3d(64 + 64 + 64, 32, bn=False, act="tanh")
        self.up3 = QRNNUpsampleConv3d(32 + 32 + 64, 16, bn=False, act="tanh")
        self.reconstructor = BiQRNNDeConv3D(16 + 16 + 64, 1, bias=True, bn=False, act="tanh")

    @staticmethod
    def _expand_ref(ref: torch.Tensor, bands: int) -> torch.Tensor:
        return ref.unsqueeze(2).expand(ref.shape[0], ref.shape[1], bands, ref.shape[2], ref.shape[3])

    def forward(
        self,
        hsi_feats: Sequence[torch.Tensor],
        ref_feats: Sequence[torch.Tensor],
        reverse: bool,
    ) -> torch.Tensor:
        bands = hsi_feats[-1].shape[2]
        x = torch.cat((hsi_feats[-1], self._expand_ref(ref_feats[-1], bands)), dim=1)
        x = self.up1(x, reverse=reverse)
        reverse = not reverse

        x = torch.cat((x, hsi_feats[-2], self._expand_ref(ref_feats[-2], bands)), dim=1)
        x = self.up2(x, reverse=reverse)
        reverse = not reverse

        x = torch.cat((x, hsi_feats[-3], self._expand_ref(ref_feats[-3], bands)), dim=1)
        x = self.up3(x, reverse=reverse)
        reverse = not reverse

        x = torch.cat((x, hsi_feats[-4], self._expand_ref(ref_feats[-4], bands)), dim=1)
        return self.reconstructor(x, reverse=reverse)


# -----------------------------------------------------------------------------
# Full HSIFN
# -----------------------------------------------------------------------------


class HSIFN(nn.Module):
    """HSIFN adapted from RGB guidance to sensor-MSI guidance."""

    def __init__(
        self,
        hsi_channels: int,
        msi_channels: int,
        srf_weights: torch.Tensor,
        *,
        use_mask: bool = True,
    ):
        super().__init__()
        hsi_channels = int(hsi_channels)
        msi_channels = int(msi_channels)
        w = torch.as_tensor(srf_weights, dtype=torch.float32)
        if w.shape != (msi_channels, hsi_channels):
            raise ValueError(
                f"SRF must be ({msi_channels},{hsi_channels}), got {tuple(w.shape)}"
            )
        self.hsi_channels = hsi_channels
        self.msi_channels = msi_channels
        self.use_mask = bool(use_mask)
        self.register_buffer("srf_weights", w.contiguous())

        self.flownet1 = FlowNet(msi_channels)
        self.flownet2 = FlowNet(msi_channels)
        self.ref_encoder = ReferenceEncoder(msi_channels)
        self.hsi_encoder = HSIEncoder(16)
        self.decoder = HSIDecoder()
        if self.use_mask:
            self.mask_predictor = MaskPredictor(64, 32, 32)

    def hsi_to_msi(self, hsi: torch.Tensor) -> torch.Tensor:
        return torch.einsum("mc,bchw->bmhw", self.srf_weights.to(hsi.dtype), hsi)

    def _align_reference(
        self,
        pseudo_msi: torch.Tensor,
        ref_hr: torch.Tensor,
    ) -> Tuple[Tuple[torch.Tensor, ...], torch.Tensor, Dict[str, torch.Tensor], List[torch.Tensor]]:
        coarse = self.flownet1(pseudo_msi, ref_hr)
        ref_coarse = flow_warp(ref_hr, coarse["flow_12_1"])

        fine = self.flownet2(pseudo_msi, ref_coarse)
        ref_feats = self.ref_encoder(ref_coarse)
        target_feats = self.ref_encoder(pseudo_msi) if self.use_mask else None
        flow_keys = ("flow_12_1", "flow_12_2", "flow_12_3", "flow_12_4")

        aligned: List[torch.Tensor] = []
        masks: List[torch.Tensor] = []
        for i, (feat, key) in enumerate(zip(ref_feats, flow_keys)):
            f = fine[key]
            if f.shape[-2:] != feat.shape[-2:]:
                f = F.interpolate(f, size=feat.shape[-2:], mode="bilinear", align_corners=True)
            warped = flow_warp(feat, f)
            if self.use_mask:
                mask = self.mask_predictor(target_feats[i], feat, f)
                warped = warped * mask
                masks.append(mask)
            aligned.append(warped)
        return tuple(aligned), ref_coarse, coarse, masks

    def forward(self, lr_hsi: torch.Tensor, ref_hr_msi: torch.Tensor):
        if lr_hsi.ndim != 4 or ref_hr_msi.ndim != 4:
            raise ValueError("expected lr_hsi/ref_hr_msi as BxCxHxW")
        if lr_hsi.shape[1] != self.hsi_channels:
            raise ValueError(f"expected {self.hsi_channels} HSI bands")
        if ref_hr_msi.shape[1] != self.msi_channels:
            raise ValueError(f"expected {self.msi_channels} MSI bands")

        h, w = ref_hr_msi.shape[-2:]
        hsi_sr = F.interpolate(lr_hsi, size=(h, w), mode="bicubic", align_corners=False)
        pseudo_msi = self.hsi_to_msi(hsi_sr)

        ref_feats, ref_coarse, coarse, masks = self._align_reference(pseudo_msi, ref_hr_msi)
        hsi_5d = hsi_sr.unsqueeze(1)  # Bx1xBandsxHxW, same convention as HSI-RefSR
        hsi_feats, reverse = self.hsi_encoder(hsi_5d)
        pred = self.decoder(hsi_feats, ref_feats, reverse).squeeze(1)

        diagnostics = {
            "pseudo_msi": pseudo_msi,
            "ref_coarse": ref_coarse,
            "coarse_flow": coarse["flow_12_1"],
            "masks": masks,
        }
        return pred, diagnostics


__all__ = ["HSIFN", "FlowNet", "flow_warp"]
