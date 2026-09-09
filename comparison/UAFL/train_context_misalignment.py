"""UAFL non-registration training with warp-before-crop HR-MSI context.

The normal shared benchmark keeps 64x64 train patches and 128x128 validation/test
regions. For misalignment training only, this entrypoint gives each train patch
a larger HR parent context, generates the MSI on that parent, applies the shared
misalignment warp there, and center-crops the warped MSI back to 64x64.

This removes artificial zero-padding caused by warping an already-cropped 64x64
MSI. GT-HSI and LR-HSI remain the original 64x64 target. Context coordinates
that would touch validation/test regions are excluded to preserve the spatial
split.
"""
from __future__ import annotations

import math
import os
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

import torch

import data_loader as shared_data
from srf_utils import hsi_to_msi_numpy


def _cli_float(name: str, default: float) -> float:
    flag = f"--{name}"
    argv = sys.argv[1:]
    value = float(default)
    for i, token in enumerate(argv):
        if token == flag and i + 1 < len(argv):
            value = float(argv[i + 1])
        elif token.startswith(flag + "="):
            value = float(token.split("=", 1)[1])
    return value


def _context_margin() -> int:
    # Current formal run is translation-only. Keep enough extra support for
    # bilinear sampling: ceil(d) pixels plus a 2-pixel interpolation guard.
    d = _cli_float("translation_max_px", 0.0)
    return int(math.ceil(max(d, 0.0))) + 2 if d > 0.0 else 0


CONTEXT_MARGIN = _context_margin()
OriginalDataset = shared_data.HSIHSRDataset


class ContextHSIHSRDataset(OriginalDataset):
    """Shared dataset with a larger MSI parent only for train split."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.misalignment_context_margin = CONTEXT_MARGIN if self.split == "train" else 0
        m = self.misalignment_context_margin
        if m <= 0:
            return

        h, w, _ = self.img.shape
        kept = []
        for top, left in self.coords:
            context_rect = (
                top - m,
                left - m,
                top + self.patch_size + m,
                left + self.patch_size + m,
            )
            if context_rect[0] < 0 or context_rect[1] < 0:
                continue
            if context_rect[2] > h or context_rect[3] > w:
                continue
            if shared_data.intersects(context_rect, self.validation_rect):
                continue
            if shared_data.intersects(context_rect, self.test_rect):
                continue
            kept.append((top, left))

        if not kept:
            raise RuntimeError(
                f"No train patches remain after applying context margin={m}px"
            )
        print(
            f"Warp-before-crop train context: margin={m}px, "
            f"parent={self.patch_size + 2*m}x{self.patch_size + 2*m}, "
            f"target={self.patch_size}x{self.patch_size}, "
            f"patches={len(self.coords)}->{len(kept)}"
        )
        self.coords = kept

    def __getitem__(self, index: int):
        m = self.misalignment_context_margin
        if self.split != "train" or m <= 0:
            return super().__getitem__(index)

        top, left = self.coords[index]
        p = self.patch_size
        parent = self.img[
            top - m : top + p + m,
            left - m : left + p + m,
            :,
        ].copy()

        # Apply exactly one geometric augmentation to the parent so its center
        # crop and the parent MSI remain pixel-aligned before misregistration.
        if self.augment:
            parent = self.random_augment(parent)

        gt = parent[m : m + p, m : m + p, :].copy()
        lr_hsi = shared_data.make_lr_hsi(
            gt,
            self.scale_ratio,
            degradation_operator=self.degradation_operator,
        )
        if self.srf_weights is not None:
            hr_msi_parent = hsi_to_msi_numpy(parent, self.srf_weights)
        else:
            hr_msi_parent = shared_data.make_hr_msi(parent, self.n_select_bands)

        return {
            "lr_hsi": shared_data.hsi_to_tensor(lr_hsi),
            # Deliberately return the larger parent under the normal key. The
            # patched prepare_reference below warps it then crops to p x p.
            "hr_msi": shared_data.hsi_to_tensor(hr_msi_parent),
            "gt": shared_data.hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }


# build_datasets resolves HSIHSRDataset from data_loader globals at call time.
shared_data.HSIHSRDataset = ContextHSIHSRDataset

import train as uafl_train  # noqa: E402

_original_prepare_reference = uafl_train.prepare_reference
_original_load_resume = uafl_train.load_resume


def _center_crop(x: torch.Tensor, size: int) -> torch.Tensor:
    h, w = x.shape[-2:]
    if h == size and w == size:
        return x
    if h < size or w < size:
        raise ValueError(f"Cannot center-crop {(h, w)} to {(size, size)}")
    top = (h - size) // 2
    left = (w - size) // 2
    return x[..., top : top + size, left : left + size]


def prepare_reference_warp_before_crop(hr_msi, args, generator=None):
    ref, valid = _original_prepare_reference(hr_msi, args, generator)
    expected_parent = int(args.patch_size) + 2 * CONTEXT_MARGIN
    # Validation/test keep their original 128x128 samples. Crop only the train
    # parent whose size matches the context construction above.
    if CONTEXT_MARGIN > 0 and tuple(hr_msi.shape[-2:]) == (expected_parent, expected_parent):
        ref = _center_crop(ref, int(args.patch_size))
        valid = _center_crop(valid, int(args.patch_size))
    return ref, valid


def load_resume_for_curriculum(model, optimizer, path, device):
    start, best = _original_load_resume(model, optimizer, path, device)
    if path and os.environ.get("UAFL_RESET_RESUME_BEST", "0") == "1":
        print(
            "Curriculum stage changed: keeping resumed model/AdamW/epoch but "
            f"resetting best_PSNR from {best:.4f} to -inf for the new d."
        )
        best = float("-inf")
    return start, best


uafl_train.prepare_reference = prepare_reference_warp_before_crop
uafl_train.load_resume = load_resume_for_curriculum


if __name__ == "__main__":
    print(
        "UAFL misalignment geometry: warp HR-MSI parent first, then center-crop "
        f"to train patch; context_margin={CONTEXT_MARGIN}px."
    )
    uafl_train.main()
