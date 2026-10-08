"""EMR-Diff backbone and exact-data adapter for Augsburg-2 strict Wald x3.

Reuses UAFL's validated WaldDataset, validity masks, radiometry and pooled
reference metrics. Only the reconstruction *method* is different.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf

HERE = Path(__file__).resolve().parent
UAFL_ROOT = HERE.parent / "UAFL"
if str(UAFL_ROOT) not in sys.path:
    sys.path.insert(0, str(UAFL_ROOT))

from augsburg2_wald_common import (  # noqa: E402
    WaldDataset, PROVENANCE, correct_msi, read_json, read_radiometry,
    require_wald, metrics_sums, metrics_from_sums,
)
from arch.BAFUnet import BAFUNet  # noqa: E402
from EMRDiff import EMRDIFF, Edge  # noqa: E402

PROTOCOL = "Augsburg-2-Wald-EMR-Diff"
HSI_BANDS = 242
MSI_BANDS = 4
STATE_BANDS = HSI_BANDS + MSI_BANDS
SCALE = 3


class EMRBackbone(nn.Module):
    """Original BAFUNet blocks with a narrow latent trunk and 1x1 state head.

    Unlike the legacy 31-band EMR implementation, 246 spectral state channels
    are not forced to be the 7x7-convolution trunk width. The architecture's
    topology, diffusion state, and intermediate supervision remain unchanged.
    """
    def __init__(self, width=64, image_size=80):
        super().__init__()
        if width < 32 or width % 32:
            raise ValueError("EMR Wald width must be >=32 and divisible by 32")
        self.width = int(width)
        self.core = BAFUNet(
            image_size=image_size,
            in_channels=STATE_BANDS,
            model_channels=width,
            out_channels=width,
            lqrgb_channels=STATE_BANDS,
            rgb_channels=MSI_BANDS,
            channel_mult=[1, 1, 1, 1, 1],
            num_res_blocks=[1, 1, 1, 1, 1],
            dims=2,
        )
        self.state_head = nn.Conv2d(width, STATE_BANDS, kernel_size=1)

    def forward(self, x_t, msi, lq_hr, t):
        latent, feature_maps = self.core(x_t, msi, lq_hr, t)
        return self.state_head(latent), [
            self.state_head(f) for f in feature_maps
        ]


def build_model(width=64, device="cpu"):
    return EMRBackbone(width=width).to(device)


def build_diffusion(device):
    cfg = OmegaConf.load(HERE / "config" / "5_step_EMRDiff.yaml")
    cfg.diffusion.params.sf = SCALE
    cfg.diffusion.params.band_dim = HSI_BANDS
    return EMRDIFF(cfg.diffusion).to(device)


def pack_batch(batch, device, calibration):
    """Honor the x3 crop first; pad only model tensors to multiples of 16."""
    gt = batch["gt"].to(device, dtype=torch.float32)
    lr = batch["lr_hsi"].to(device, dtype=torch.float32)
    msi = correct_msi(batch["hr_msi"].to(device, dtype=torch.float32), calibration)
    mask = batch["valid_mask"].to(device, dtype=torch.float32)
    if gt.ndim != 4 or gt.shape[1] != HSI_BANDS:
        raise ValueError("Strict Wald requires Bx242xHxW observed 30m HSI")
    if msi.shape[1] != MSI_BANDS or gt.shape[-2:] != msi.shape[-2:]:
        raise ValueError("Strict Wald requires 4 measured Sentinel-2 MSI bands")
    h, w = gt.shape[-2:]
    if lr.shape[-2:] != (h // SCALE, w // SCALE) or h % SCALE or w % SCALE:
        raise ValueError("Wald LR-HSI must be exactly x3 on the observed grid")
    lr_hr = F.interpolate(lr, size=(h, w), mode="bicubic", align_corners=False)
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        pad = (0, pw, 0, ph)
        gt = F.pad(gt, pad, mode="replicate")
        msi = F.pad(msi, pad, mode="replicate")
        lr_hr = F.pad(lr_hr, pad, mode="replicate")
        mask = F.pad(mask, pad, value=0.0)
    return gt, lr_hr, msi, mask, (h, w)


def masked_l1(pred, target, mask):
    if pred.shape != target.shape:
        raise ValueError("EMR multiscale target shape mismatch")
    good = (mask > 0.5).expand_as(pred)
    if not bool(good.any()):
        raise ValueError("No valid Wald pixels")
    return (pred - target).abs()[good].mean()


def down_to(x, hw):
    h, w = x.shape[-2:]
    th, tw = hw
    if (h, w) == (th, tw):
        return x
    if th <= h and tw <= w and h % th == 0 and w % tw == 0:
        return x[..., :: h // th, :: w // tw][..., :th, :tw]
    return F.interpolate(x, size=(th, tw), mode="bicubic", align_corners=False)


def training_step(model, diffusion, edge, gt, lr_hr, msi, mask):
    condition = torch.cat((lr_hr, msi), dim=1)
    x_start = torch.cat((gt, gt[:, :MSI_BANDS]), dim=1)
    t = torch.randint(0, diffusion.num_diffusion_timesteps,
                      (gt.shape[0],), device=gt.device)
    noise = torch.randn_like(condition)
    x_t = diffusion.forward_addnoise(x_start, condition, t, noise, rgb_hr=msi)
    residual, intermediate = model(x_t, msi, lr_hr, t)
    loss = masked_l1(residual + condition, x_start, mask)
    # Preserve EMR-Diff multiscale residual supervision (layers 2, 4, 6).
    for idx in (2, 4, 6):
        if idx >= len(intermediate):
            continue
        f = intermediate[idx]
        hw = f.shape[-2:]
        sub_mask = F.interpolate(mask, size=hw, mode="nearest")
        loss = loss + masked_l1(
            f + down_to(condition, hw),
            down_to(x_start, hw),
            sub_mask,
        )
    return loss


@torch.no_grad()
def predict(model, diffusion, edge, lr_hr, msi):
    model.eval()
    cond = torch.cat((lr_hr, msi), dim=1)
    emap = edge(msi)
    x_t = diffusion.prior_sample(cond, torch.randn_like(cond), edge_map=emap)
    for step in range(diffusion.num_diffusion_timesteps - 1, -1, -1):
        t = torch.full((cond.shape[0],), step,
                       device=cond.device, dtype=torch.long)
        residual, _ = model(x_t, msi, lr_hr, t)
        start = residual + cond
        x_t = diffusion.inverse_denoise(
            x_start=start, x_t=x_t, t=t,
            noise=torch.randn_like(start), edge_map=emap,
        )
    return x_t[:, :HSI_BANDS]


@torch.no_grad()
def predict_batch(model, diffusion, edge, batch, device, calibration):
    gt, lr_hr, msi, mask, (h, w) = pack_batch(batch, device, calibration)
    return predict(model, diffusion, edge, lr_hr, msi)[..., :h, :w], (
        gt[..., :h, :w], mask[..., :h, :w]
    )


def verify_checkpoint(state, *, radiometry_sha, sigma, monitor=None, width=None):
    if (state.get("protocol") != PROTOCOL
            or state.get("msi_source") != PROVENANCE
            or state.get("target") != "observed_30m_HSI_only"
            or state.get("scale_ratio") != SCALE
            or state.get("radiometry_sha256") != radiometry_sha
            or abs(float(state.get("wald_sigma", -999)) - sigma) > 1e-8):
        raise ValueError("EMR checkpoint is not from this strict Wald calibration/protocol")
    if monitor is not None and state.get("monitor") != monitor:
        raise ValueError("Checkpoint monitor differs; do not mix PSNR/SAM selections")
    if width is not None and int(state.get("model_width", -1)) != width:
        raise ValueError("Checkpoint hidden width differs")
