"""Strict Augsburg-2 Wald paired cache adapter for UAFL.

Matches S2Diff-MH's prepare_augsburg2_wald.py and AugsburgRealDataset for
the strict Wald branch. Never load EnMAP10 or synthesize a new observation.
"""
from __future__ import annotations

import json
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


PROVENANCE = "real_Sentinel_2_Wald_30m"


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def require_wald(wald_root):
    root = Path(wald_root)
    for split in ("train", "validation", "test"):
        meta = read_json(root / split / "meta.json")
        if (meta.get("msi_source") != PROVENANCE or
            meta.get("target") != "30m_EnMAP_like" or
            meta.get("gt_source") != "observed_30m_HSI_only" or
            int(meta.get("scale_ratio", -1)) != 3):
            raise ValueError(f"{split} is not strict Augsburg-2 Wald cache: {meta}")
    psf = read_json(root / "wald_psf.json")
    if int(psf.get("scale_ratio", -1)) != 3:
        raise ValueError("Expected Wald x3 operator metadata")
    srf = np.load(root / "srf_weights.npy")
    if srf.shape != (4, 242) or not np.isfinite(srf).all():
        raise ValueError(f"Invalid observed sensor SRF shape: {srf.shape}")
    full = read_json(root / "full" / "meta.json")
    if full.get("region") != "sub_area_2":
        raise ValueError("Strict Wald full inference must use sub_area_2")
    return float(psf["terminal_sigma_hr_pixels"])


def read_radiometry(path):
    data = read_json(path)
    if data.get("dataset") != "Augsburg-2-Wald" or data.get("uses_EnMAP10_reference") is not False:
        raise ValueError("Require train-only Augsburg2_Wald_radiometry.json (no EnMAP10)")
    gain, bias = np.asarray(data["gain"], dtype=np.float32), np.asarray(data["bias"], dtype=np.float32)
    if gain.shape != (4,) or bias.shape != (4,) or not np.isfinite(gain).all() or not np.isfinite(bias).all():
        raise ValueError("Radiometry requires 4 finite gains and biases")
    return gain, bias


def correct_msi(x, calibration):
    gain, bias = calibration
    g = torch.as_tensor(gain, device=x.device, dtype=x.dtype).view(1, 4, 1, 1)
    b = torch.as_tensor(bias, device=x.device, dtype=x.dtype).view(1, 4, 1, 1)
    return x * g + b


def upsample(lr, size):
    return F.interpolate(lr, size=size, mode="bicubic", align_corners=False)


def predict_uafl(model, lr_hsi, hr_msi):
    """Retain observed Wald spatial crop, pad only the network's window input.

    UAFL has window size 8 at HR and half-resolution. Thus the internal
    network input H/W must be multiples of 16. S2Diff-MH Wald trains on
    72x72 HR crops (half-resolution 36, invalid for UAFL window partition).
    Reflect the border to 80x80 inside the network only, then crop back to
    72x72 before masked loss/metrics. No new observations are synthesized.
    """
    h, w = hr_msi.shape[-2:]
    x = upsample(lr_hsi, (h, w))
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        ref = F.pad(hr_msi, (0, pw, 0, ph), mode="replicate")
    else:
        ref = hr_msi
    return model(x, ref)[..., :h, :w]


def grid_coords(h, w, patch, stride):
    return [(i, j, patch, patch)
            for i in range(0, h - patch + 1, stride)
            for j in range(0, w - patch + 1, stride)]


def partition_tiles(h, w, size):
    if h % 3 or w % 3 or size % 3:
        raise ValueError("Wald evaluation tiles must respect native x3 grid")
    return [(top, left, min(size, h - top), min(size, w - left))
            for top in range(0, h, size)
            for left in range(0, w, size)]


class WaldDataset(Dataset):
    def __init__(self, wald_root, split, train_patch=72, train_stride=6,
                 eval_patch=48, min_valid_fraction=0.80):
        self.split = split
        root = Path(wald_root) / split
        metadata = read_json(root / "meta.json")
        self.forbidden_bbox = (
            metadata.get("forbidden_bbox_30m") if split == "train" else None
        )
        self.gt = np.load(root / "gt.npy", mmap_mode="r")
        self.lr = np.load(root / "lr_hsi.npy", mmap_mode="r")
        self.msi = np.load(root / "hr_msi.npy", mmap_mode="r")
        self.mask = np.load(root / "valid_mask.npy", mmap_mode="r")
        h, w, bands = self.gt.shape
        if (bands != 242 or self.msi.shape != (h, w, 4) or
            self.lr.shape != (h // 3, w // 3, 242) or
            self.mask.shape != (h, w) or h % 3 or w % 3):
            raise ValueError(f"Invalid Wald array shapes in {root}")
        p = train_patch if split == "train" else eval_patch
        stride = train_stride if split == "train" else eval_patch
        if p % 24 or stride % 3:
            raise ValueError("UAFL Wald patch must be divisible by 24; stride by 3")
        candidates = (grid_coords(h, w, p, stride) if split == "train"
                      else partition_tiles(h, w, p))
        self.tiles = []
        for top, left, ph, pw in candidates:
            if self.forbidden_bbox is not None:
                fy0, fx0, fy1, fx1 = map(int, self.forbidden_bbox)
                if top < fy1 and top + ph > fy0 and left < fx1 and left + pw > fx0:
                    continue
            if float(np.asarray(self.mask[top:top+ph, left:left+pw]).mean()) >= min_valid_fraction:
                self.tiles.append((top, left, ph, pw))
        if not self.tiles:
            raise ValueError(f"No valid {split} tiles at min_valid_fraction={min_valid_fraction}")
        # Strict S2Diff-MH Wald training disables spatial augmentation;
        # the original observed HSI / MSI misregistration is preserved.

    def __len__(self):
        return len(self.tiles)

    def __getitem__(self, idx):
        top, left, ph, pw = self.tiles[idx]
        gt = np.asarray(self.gt[top:top+ph, left:left+pw]).copy()
        msi = np.asarray(self.msi[top:top+ph, left:left+pw]).copy()
        lr = np.asarray(self.lr[top//3:(top+ph)//3, left//3:(left+pw)//3]).copy()
        mask = np.asarray(self.mask[top:top+ph, left:left+pw]).copy().astype(np.float32)
        if (ph % 8 or pw % 8):
            # UAFL attention uses 8-divisible spatial dimensions. Pad only
            # eval edge tiles; padded pixels never enter the metric.
            pad_h, pad_w = (-ph) % 24, (-pw) % 24
            gt = np.pad(gt, ((0,pad_h),(0,pad_w),(0,0)), mode="edge")
            msi = np.pad(msi, ((0,pad_h),(0,pad_w),(0,0)), mode="edge")
            lr = np.pad(lr, ((0,pad_h//3),(0,pad_w//3),(0,0)), mode="edge")
            mask = np.pad(mask, ((0,pad_h),(0,pad_w)), constant_values=0)
        def chw(value):
            return torch.from_numpy(np.ascontiguousarray(value.transpose(2,0,1))).float()
        return {
            "gt":chw(gt), "hr_msi":chw(msi), "lr_hsi":chw(lr),
            "valid_mask":torch.from_numpy(np.ascontiguousarray(mask[None])).float(),
        }


def make_loaders(root, *, batch_size=1, train_patch=72, train_stride=6,
                 eval_patch=48, min_valid_fraction=0.8, workers=0,
                 include_test=False):
    # Training must never open the held-out test split.
    splits = ("train", "validation", "test") if include_test else ("train", "validation")
    sets = [WaldDataset(root, split, train_patch, train_stride, eval_patch, min_valid_fraction)
            for split in splits]
    loaders = [DataLoader(ds, batch_size=batch_size if i==0 else 1,
                          shuffle=i==0, num_workers=workers, drop_last=False)
               for i, ds in enumerate(sets)]
    if include_test:
        return (*loaders, [len(ds) for ds in sets])
    return (loaders[0], loaders[1], None, [len(ds) for ds in sets])


def masked_l1(pred, gt, mask):
    good = mask.expand_as(pred) > 0.5
    if not bool(good.any()):
        raise ValueError("empty valid patch")
    return (pred - gt).abs()[good].mean()


def metrics_sums(pred, target, mask):
    good = mask.expand_as(pred) > 0.5
    diff = (pred.float()-target.float())[good].double()
    sse = float((diff*diff).sum().item())
    n = int(diff.numel())
    pn = torch.linalg.vector_norm(pred.float(), dim=1)
    tn = torch.linalg.vector_norm(target.float(), dim=1)
    valid = (mask[:,0] > .5) & (pn > 1e-12) & (tn > 1e-12)
    if bool(valid.any()):
        dots = (pred.float()*target.float()).sum(1)
        cos = (dots[valid]/(pn[valid]*tn[valid]).clamp_min(1e-12)).clamp(-1,1)
        angles = torch.acos(cos).double()
        return sse, n, float(angles.sum().item()), int(angles.numel())
    return sse, n, 0., 0


def metrics_from_sums(sums):
    sse, n, angle_sum, n_angle = sums
    if not n:
        raise ValueError("no valid evaluation pixels")
    return {
        "ref_psnr":-10*math.log10(max(sse/n,1e-12)),
        "ref_sam":angle_sum/max(n_angle,1)*180/math.pi if n_angle else float("nan"),
        "ref_rmse":math.sqrt(sse/n),
    }


def load_state(path, model, optimizer=None, device="cpu"):
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    if checkpoint.get("protocol") != "Augsburg-2-Wald-UAFL" or checkpoint.get("msi_source") != PROVENANCE:
        raise ValueError("Checkpoint must be a strictly Wald-trained UAFL checkpoint")
    model.load_state_dict(checkpoint["model"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint
