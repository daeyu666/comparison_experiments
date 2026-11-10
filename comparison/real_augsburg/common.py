"""Inputs, provenance and output writers for Augsburg Region 2."""
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from comparison.UAFL.augsburg2_wald_center_roi import geotiff_transform, read_roi
from comparison.UAFL.augsburg2_wald_common import read_json, read_radiometry, require_wald
from experiment_protocol import (
    REAL_FORBIDDEN_BBOX_30M, REAL_SPLIT_PROTOCOL, REAL_TEST_BBOX_10M, REAL_TEST_BBOX_30M,
)


@dataclass
class WaldCache:
    root: Path
    radiometry: Path
    calibration: tuple
    sigma: float
    srf: np.ndarray
    fingerprint: dict

    @classmethod
    def load(cls, wald_root, radiometry_json):
        root, rad = Path(wald_root), Path(radiometry_json)
        sigma = require_wald(root)
        roi = read_roi(root)
        if roi is None:
            raise ValueError("The center-heldout Wald cache must contain roi.json")
        expected = {
            "protocol_id": REAL_SPLIT_PROTOCOL,
            "test_bbox_30m": list(REAL_TEST_BBOX_30M),
            "test_bbox_10m": list(REAL_TEST_BBOX_10M),
            "forbidden_bbox_30m": list(REAL_FORBIDDEN_BBOX_30M),
        }
        if any(roi.get(k) != v for k, v in expected.items()):
            raise ValueError("Augsburg center ROI or training guard differs from the shared protocol")
        for split in ("train", "validation", "test"):
            meta = read_json(root / split / "meta.json")
            if meta.get("protocol_id") != REAL_SPLIT_PROTOCOL:
                raise ValueError(f"{split} metadata belongs to a different Wald split")
            if split != "validation" and meta.get("test_bbox_30m") != expected["test_bbox_30m"]:
                raise ValueError(f"{split} metadata has incorrect holdout ROI")
            if split == "train" and meta.get("forbidden_bbox_30m") != expected["forbidden_bbox_30m"]:
                raise ValueError("The training split does not enforce the center ROI and PSF guard")
        if not np.isfinite(sigma) or sigma < 0:
            raise ValueError("Wald sigma must be finite and nonnegative")
        calib = read_radiometry(rad, root)
        srf_file = root / "srf_weights.npy"
        return cls(root, rad, calib, sigma, np.load(srf_file), {
            "split_protocol_id": REAL_SPLIT_PROTOCOL, "scale_ratio": 3,
            "test_bbox_30m": expected["test_bbox_30m"],
            "forbidden_bbox_30m": expected["forbidden_bbox_30m"],
            "radiometry_sha256": sha256(rad.read_bytes()).hexdigest(),
            "srf_sha256": sha256(srf_file.read_bytes()).hexdigest(),
            "wald_sigma": sigma, "hsi_channels": 242, "msi_channels": 4,
        })

    def native_inputs(self):
        full = self.root / "full"
        lr, msi, mask = (np.load(full / f"{name}.npy", mmap_mode="r")
                         for name in ("lr_hsi", "hr_msi", "valid_mask"))
        if (lr.shape != (100, 120, 242) or msi.shape != (300, 360, 4)
                or mask.shape != (300, 360)):
            raise ValueError(f"Unexpected Region 2 array shapes: {lr.shape}, {msi.shape}, {mask.shape}")
        y, x, yy, xx = REAL_TEST_BBOX_30M
        Y, X, YY, XX = REAL_TEST_BBOX_10M
        return lr[y:yy, x:xx], msi[Y:YY, X:XX], mask[Y:YY, X:XX], read_json(full / "meta.json")


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def write_geotiff(path, cube, meta):
    import rasterio

    y, x, *_ = REAL_TEST_BBOX_10M
    h, w, channels = cube.shape
    with rasterio.open(
        path, "w", driver="GTiff", height=h, width=w, count=channels, dtype="float32",
        crs=meta["crs"], transform=geotiff_transform(meta["transform_6"], y, x),
        compress="deflate", tiled=True,
    ) as dest:
        dest.write(cube.transpose(2, 0, 1))


def chw(array, device):
    return torch.from_numpy(np.array(array.transpose(2, 0, 1), dtype=np.float32, order="C"))[None].to(device)


def positions(length, tile, stride):
    if not 0 < stride <= tile <= length or any(n % 3 for n in (length, tile, stride)):
        raise ValueError("Native tile and stride must align with the x3 grid")
    values = list(range(0, length - tile + 1, stride))
    if values[-1] != length - tile:
        values.append(length - tile)
    return values


def pad_pair(lr, msi, multiple, minimum=0):
    """Pad the LR grid and its exact x3 MSI footprint; preserve all observed samples."""
    h, w = lr.shape[-2:]
    ph, pw = (max(minimum, n + (-n) % multiple) - n for n in (h, w))
    if ph or pw:
        lr = F.pad(lr, (0, pw, 0, ph), mode="replicate")
        msi = F.pad(msi, (0, 3 * pw, 0, 3 * ph), mode="replicate")
    return lr, msi
