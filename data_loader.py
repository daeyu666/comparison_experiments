"""Unified comparison-experiment data loader.

This file mirrors the benchmark splits and sensor protocols in S2Diff-MH while
also generating LR-HSI with the comparison repository's shared degradation
operator.

Frozen benchmark splits:
- PaviaU: center 128x128 test; top-left 128x128 validation; remainder train.
- Houston2013: same center128 protocol.
- Chikusei: center-crop 2304x2048; top 128-row strip test (16x128 patches),
  next 128-row strip validation (16 patches), rows 256:2304 train.
- CAVE: deterministic 16 train / 4 validation / 12 test scenes.
- Botswana: center128 protocol.
- Augsburg synthetic x4: official MDAS geographic train/validation/test files.

All comparison methods should consume these loaders or exactly reproduce these
machine-readable splits under data/splits/.
"""

from __future__ import annotations

import functools
import glob
import os
import random
import re
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from degradations import build_degradation
from srf_utils import (
    EO1_ALI_8_BANDS,
    IKONOS_4_BANDS,
    NIKON_D700_3_BANDS,
    S2A_NATIVE10_4_BANDS,
    WV2_ALL_8_BANDS,
    WV2_VISIBLE_5_BANDS,
    WV2_VISIBLE_6_BANDS,
    build_srf_weights,
    hsi_to_msi_numpy,
    load_hsi_wavelengths,
    print_srf_summary,
    sensor_protocol,
)

try:
    import scipy.io as scio
except ImportError:
    scio = None
try:
    import hdf5storage
except ImportError:
    hdf5storage = None
try:
    import h5py
except ImportError:
    h5py = None


CAVE_TRAIN_SCENES = [
    "balloons", "beads", "cd", "chart_and_stuffed_toy", "clay", "cloth",
    "egyptian_statue", "face", "fake_and_real_beers", "fake_and_real_food",
    "fake_and_real_lemon_slices", "fake_and_real_lemons",
    "fake_and_real_peppers", "fake_and_real_strawberries",
    "fake_and_real_sushi", "fake_and_real_tomatoes",
]
CAVE_VALIDATION_SCENES = ["feathers", "flowers", "glass_tiles", "hairs"]
CAVE_TEST_SCENES = [
    "jelly_beans", "oil_painting", "paints", "photo_and_face", "pompoms",
    "real_and_fake_apples", "real_and_fake_peppers", "sponges", "stuffed_toys",
    "superballs", "thread_spools", "watercolors",
]


def read_hsi_mat(file_path: str, candidate_keys: Sequence[str]) -> np.ndarray:
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Cannot find data file: {file_path}")
    mat_data = None
    if hdf5storage is not None:
        try:
            mat_data = hdf5storage.loadmat(file_path)
        except Exception:
            mat_data = None
    if mat_data is None and scio is not None:
        try:
            mat_data = scio.loadmat(file_path)
        except Exception:
            mat_data = None
    if mat_data is not None:
        for key in list(candidate_keys) + list(mat_data.keys()):
            if key in mat_data and isinstance(mat_data[key], np.ndarray):
                arr = np.asarray(mat_data[key]).squeeze()
                if arr.ndim == 3:
                    return fix_hsi_shape(arr)
    if h5py is not None:
        with h5py.File(file_path, "r") as f:
            for key in list(candidate_keys) + list(f.keys()):
                if key in f:
                    arr = np.asarray(f[key]).squeeze()
                    if arr.ndim == 3:
                        return fix_hsi_shape(arr)
    raise RuntimeError(f"No valid 3-D HSI array found in {file_path}")


def fix_hsi_shape(img: np.ndarray, expected_bands: int | None = None) -> np.ndarray:
    img = np.asarray(img).squeeze()
    if img.ndim != 3:
        raise ValueError(f"HSI data must be 3-D, got {img.shape}")
    if expected_bands is not None:
        axes = [i for i, size in enumerate(img.shape) if size == int(expected_bands)]
        if len(axes) == 1 and axes[0] != 2:
            img = np.moveaxis(img, axes[0], 2)
            return img.astype(np.float32)
    if img.shape[0] <= 256 and img.shape[1] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[1] <= 256 and img.shape[0] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (0, 2, 1))
    return img.astype(np.float32)


def normalize_hsi(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    lo, hi = float(np.nanmin(img)), float(np.nanmax(img))
    if hi - lo < 1e-8:
        return np.zeros_like(img)
    return np.clip((img - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    h, w, _ = img.shape
    return img[:h // scale_ratio * scale_ratio, :w // scale_ratio * scale_ratio, :]


def hsi_to_tensor(img: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def tensor_to_hsi(x: torch.Tensor) -> np.ndarray:
    if x.ndim != 3:
        raise ValueError(f"Expected CxHxW tensor, got {tuple(x.shape)}")
    return x.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def build_hsi_degradation(cfg):
    mode = getattr(cfg, "degradation_mode", "gaussian_bicubic")
    if mode == "gaussian_bicubic":
        return build_degradation(
            mode,
            scale_ratio=cfg.scale_ratio,
            sigma=getattr(cfg, "degradation_sigma", 2.0),
            kernel_size=getattr(cfg, "degradation_kernel_size", 5),
        )
    if mode == "physical":
        return build_degradation(
            mode,
            scale_ratio=cfg.scale_ratio,
            mtf_nyquist=getattr(cfg, "mtf_nyquist", 0.2),
            truncate=getattr(cfg, "psf_truncate", 3.0),
        )
    raise ValueError(f"Unsupported degradation_mode={mode!r}")


def make_lr_hsi(hr_hsi: np.ndarray, scale_ratio: int, degradation_operator=None) -> np.ndarray:
    if degradation_operator is None:
        degradation_operator = build_degradation(
            "gaussian_bicubic", scale_ratio=scale_ratio, sigma=2.0, kernel_size=5
        )
    x = hsi_to_tensor(hr_hsi).unsqueeze(0)
    with torch.no_grad():
        y = degradation_operator.degrade(x).squeeze(0)
    return tensor_to_hsi(y)


def make_hr_msi(hr_hsi: np.ndarray, n_select_bands: int) -> np.ndarray:
    n_bands = hr_hsi.shape[2]
    idx = np.linspace(0, n_bands - 1, n_select_bands).round().astype(np.int64)
    return hr_hsi[:, :, idx].astype(np.float32)


def _center_rect(h: int, w: int, size: int) -> Tuple[int, int, int, int]:
    if h < size or w < size:
        raise ValueError(f"Image {(h,w)} smaller than {size}x{size}")
    top = (h - size) // 2
    left = (w - size) // 2
    return top, left, top + size, left + size


def _intersects(a, b) -> bool:
    t1, l1, b1, r1 = a
    t2, l2, b2, r2 = b
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def _grid_coords(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    return [
        (top, left)
        for top in range(0, h - patch + 1, stride)
        for left in range(0, w - patch + 1, stride)
    ]


def _single_scene_split(h, w, patch_size, stride, split, test_size):
    test_rect = _center_rect(h, w, test_size)
    val_rect = (0, 0, test_size, test_size)
    if _intersects(val_rect, test_rect):
        val_rect = (0, w - test_size, test_size, w)
    if split == "test":
        return [(test_rect[0], test_rect[1])], val_rect, test_rect
    if split in ("validation", "val"):
        return [(val_rect[0], val_rect[1])], val_rect, test_rect
    coords = []
    for top, left in _grid_coords(h, w, patch_size, stride):
        rect = (top, left, top + patch_size, left + patch_size)
        if not _intersects(rect, val_rect) and not _intersects(rect, test_rect):
            coords.append((top, left))
    if not coords:
        raise RuntimeError("No training patches remain after split exclusion")
    return coords, val_rect, test_rect


def _center_crop(img: np.ndarray, th: int, tw: int) -> np.ndarray:
    h, w, _ = img.shape
    top, left = (h - th) // 2, (w - tw) // 2
    if top < 0 or left < 0:
        raise ValueError(f"Cannot center-crop {(h,w)} to {(th,tw)}")
    return img[top:top+th, left:left+tw, :]


def _chikusei_coords(patch_size, stride, split, test_size):
    if test_size != 128:
        raise ValueError("Fixed Chikusei benchmark uses image_size=128")
    if split == "test":
        return [(0, x) for x in range(0, 2048, 128)]
    if split in ("validation", "val"):
        return [(128, x) for x in range(0, 2048, 128)]
    return [
        (y, x)
        for y in range(256, 2304 - patch_size + 1, stride)
        for x in range(0, 2048 - patch_size + 1, stride)
    ]


def _uniform_or_srf_msi(gt, srf_weights, n_select_bands):
    return hsi_to_msi_numpy(gt, srf_weights) if srf_weights is not None else make_hr_msi(gt, n_select_bands)


class HSIHSRDataset(Dataset):
    def __init__(
        self,
        img,
        dataset_name,
        patch_size,
        coords,
        scale_ratio,
        n_select_bands,
        split,
        augment,
        srf_weights,
        degradation_operator,
    ):
        self.img = img
        self.dataset_name = dataset_name
        self.patch_size = int(patch_size)
        self.coords = list(coords)
        self.scale_ratio = int(scale_ratio)
        self.n_select_bands = int(n_select_bands)
        self.split = split
        self.augment = bool(augment and split == "train")
        self.srf_weights = srf_weights
        self.degradation_operator = degradation_operator

    def __len__(self):
        return len(self.coords)

    def __getitem__(self, index):
        top, left = self.coords[index]
        gt = self.img[top:top+self.patch_size, left:left+self.patch_size, :].copy()
        if self.augment:
            if random.random() < 0.5:
                gt = np.flip(gt, axis=0)
            if random.random() < 0.5:
                gt = np.flip(gt, axis=1)
            if random.random() < 0.5:
                gt = np.rot90(gt, k=random.randint(1,3), axes=(0,1))
            gt = np.ascontiguousarray(gt)
        lr_hsi = make_lr_hsi(gt, self.scale_ratio, self.degradation_operator)
        hr_msi = _uniform_or_srf_msi(gt, self.srf_weights, self.n_select_bands)
        return {
            "lr_hsi": hsi_to_tensor(lr_hsi),
            "hr_msi": hsi_to_tensor(hr_msi),
            "gt": hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }


def _canonical_scene_name(path: str) -> str:
    return re.sub(r"_ms$", "", os.path.basename(os.path.normpath(path)).lower())


def _find_cave_scene_dirs(root: str) -> Dict[str, str]:
    dirs = set()
    for pat in ("**/*_ms_01.png", "**/*_ms_01.PNG"):
        for f in glob.glob(os.path.join(root, pat), recursive=True):
            dirs.add(os.path.dirname(f))
    mapping = {_canonical_scene_name(p): p for p in sorted(dirs)}
    required = set(CAVE_TRAIN_SCENES + CAVE_VALIDATION_SCENES + CAVE_TEST_SCENES)
    missing = sorted(required - set(mapping))
    if missing:
        raise FileNotFoundError("Missing CAVE scenes: " + ", ".join(missing))
    return mapping


@functools.lru_cache(maxsize=4)
def _load_cave_scene(scene_dir: str) -> np.ndarray:
    try:
        from PIL import Image
    except ImportError as exc:
        raise ImportError("CAVE PNG loading requires Pillow") from exc
    bands = []
    for i in range(1, 32):
        m = glob.glob(os.path.join(scene_dir, f"*_ms_{i:02d}.png"))
        m += glob.glob(os.path.join(scene_dir, f"*_ms_{i:02d}.PNG"))
        if not m:
            raise FileNotFoundError(f"Missing CAVE band {i:02d} in {scene_dir}")
        bands.append(np.asarray(Image.open(sorted(m)[0]), dtype=np.float32))
    cube = np.stack(bands, axis=2)
    if cube.max() > 1:
        cube /= 65535.0
    return np.clip(cube, 0, 1).astype(np.float32)


class CAVEDataset(Dataset):
    def __init__(
        self, scene_dirs, scene_names, split, patch_size, stride,
        scale_ratio, n_select_bands, srf_weights, degradation_operator, augment
    ):
        self.scene_dirs = scene_dirs
        self.split = split
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.srf_weights = srf_weights
        self.degradation_operator = degradation_operator
        self.augment = bool(augment and split == "train")
        self.samples = []
        for name in scene_names:
            if split == "train":
                for top, left in _grid_coords(512, 512, patch_size, stride):
                    self.samples.append((name, top, left, patch_size))
            else:
                self.samples.append((name, 0, 0, 512))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        name, top, left, size = self.samples[index]
        gt = _load_cave_scene(self.scene_dirs[name])[top:top+size,left:left+size].copy()
        if self.augment:
            if random.random() < .5: gt = np.flip(gt, 0)
            if random.random() < .5: gt = np.flip(gt, 1)
            if random.random() < .5: gt = np.rot90(gt, random.randint(1,3), (0,1))
            gt = np.ascontiguousarray(gt)
        lr_hsi = make_lr_hsi(gt, self.scale_ratio, self.degradation_operator)
        hr_msi = _uniform_or_srf_msi(gt, self.srf_weights, self.n_select_bands)
        return {
            "lr_hsi": hsi_to_tensor(lr_hsi),
            "hr_msi": hsi_to_tensor(hr_msi),
            "gt": hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(31, dtype=torch.long),
        }


def _read_envi_wavelengths(path: str) -> np.ndarray:
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    m = re.search(r"wavelength\s*=\s*\{([^}]*)\}", text, re.I | re.S)
    if not m:
        raise ValueError(f"No wavelength field in {path}")
    values = np.asarray(
        [float(x) for x in re.findall(r"[-+]?\d*\.?\d+(?:[Ee][-+]?\d+)?", m.group(1))],
        dtype=np.float32,
    )
    if values.size != 242:
        raise ValueError(f"Expected 242 EnMAP wavelengths, got {values.size}")
    if values.max() < 10:
        values *= 1000
    return values


def _read_tiff_cube(path: str) -> np.ndarray:
    try:
        import tifffile
    except ImportError as exc:
        raise ImportError("Augsburg loading requires tifffile") from exc
    arr = fix_hsi_shape(tifffile.imread(path), expected_bands=242)
    if np.nanmax(arr) > 2:
        arr = arr / 10000.0
    return np.clip(np.nan_to_num(arr), 0, 1).astype(np.float32)


def _find_augsburg_root(data_root: str) -> str:
    candidates = [
        os.path.join(data_root, "Augsburg", "Augsburg_data_4_publication"),
        os.path.join(data_root, "Augsburg"),
        os.path.join(data_root, "Augsburg_data_4_publication"),
        data_root,
    ]
    for root in candidates:
        if os.path.exists(os.path.join(root, "band_242_meta_info.hdr")):
            return root
    raise FileNotFoundError("Cannot locate Augsburg band_242_meta_info.hdr")


def _resolve_srf_spec(cfg, n_bands: int, hsi_wavelengths_override=None):
    requested = getattr(cfg, "srf_band_set", "auto")
    if requested == "auto":
        protocol = sensor_protocol(cfg.dataset)
        selected_bands = protocol["bands"]
        srf_path = getattr(cfg, "srf_path", "") or protocol["srf_path"]
        if hsi_wavelengths_override is not None:
            wavelengths = np.asarray(hsi_wavelengths_override, dtype=np.float32)
            wavelength_path = "dataset-metadata"
        else:
            wavelength_path = getattr(cfg, "wavelength_path", "") or protocol["wavelength_path"]
            wavelengths = load_hsi_wavelengths(wavelength_path, n_bands)
        resolved = {
            "PaviaU":"ikonos4", "Houston13":"wv2_all8", "Chikusei":"wv2_all8",
            "CAVE":"nikon_d700", "Botswana":"eo1_ali8", "Augsburg":"s2a_native10_4",
        }[cfg.dataset]
        return srf_path, selected_bands, wavelengths, wavelength_path, resolved

    mapping = {
        "ikonos4": (IKONOS_4_BANDS, "./data/srf/ikonos_relative_spectral_response.csv"),
        "wv2_visible5": (WV2_VISIBLE_5_BANDS, "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"),
        "wv2_visible6": (WV2_VISIBLE_6_BANDS, "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"),
        "wv2_all8": (WV2_ALL_8_BANDS, "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"),
        "nikon_d700": (NIKON_D700_3_BANDS, "./data/srf/nikon_d700_relative_spectral_response.csv"),
        "eo1_ali8": (EO1_ALI_8_BANDS, "./data/srf/eo1_ali_8band_relative_spectral_response.csv"),
        "s2a_native10_4": (S2A_NATIVE10_4_BANDS, "./data/srf/sentinel2a_srf_v4_B2_B3_B4_B8.csv"),
    }
    if requested not in mapping:
        raise ValueError(f"Unsupported srf_band_set={requested}")
    selected_bands, default_srf = mapping[requested]
    srf_path = getattr(cfg, "srf_path", "") or default_srf
    if hsi_wavelengths_override is not None:
        wavelengths = np.asarray(hsi_wavelengths_override, dtype=np.float32)
        wavelength_path = "dataset-metadata"
    else:
        wavelength_path = getattr(cfg, "wavelength_path", "")
        if not wavelength_path:
            protocol = sensor_protocol(cfg.dataset)
            wavelength_path = protocol["wavelength_path"]
        wavelengths = load_hsi_wavelengths(wavelength_path, n_bands)
    return srf_path, selected_bands, wavelengths, wavelength_path, requested


def _make_srf(cfg, n_bands, wavelengths_override=None):
    if getattr(cfg, "msi_mode", "srf") != "srf":
        return None, None, None, None, "uniform", int(cfg.n_select_bands)
    srf_path, bands, wavelengths, wavelength_path, profile = _resolve_srf_spec(
        cfg, n_bands, wavelengths_override
    )
    weights, names = build_srf_weights(
        srf_path, wavelengths, bands, interp_kind=cfg.srf_interp, normalize=True
    )
    print_srf_summary(weights, names, wavelengths)
    return weights, names, wavelengths, wavelength_path, profile, int(weights.shape[0])


def _make_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=True, drop_last=drop_last
    )


def _standard_single_scene(cfg, img, degradation_operator):
    img = normalize_hsi(img)
    img = crop_to_scale(img, cfg.scale_ratio)
    if cfg.dataset == "Chikusei":
        img = _center_crop(img, 2304, 2048)
    weights, names, wavelengths, wavelength_path, profile, n_select = _make_srf(
        cfg, img.shape[2]
    )
    if cfg.dataset == "Chikusei":
        train_coords = _chikusei_coords(cfg.patch_size, cfg.stride, "train", cfg.image_size)
        val_coords = _chikusei_coords(cfg.image_size, cfg.image_size, "validation", cfg.image_size)
        test_coords = _chikusei_coords(cfg.image_size, cfg.image_size, "test", cfg.image_size)
        val_rect, test_rect = (128,0,256,2048), (0,0,128,2048)
    else:
        train_coords, val_rect, test_rect = _single_scene_split(
            img.shape[0], img.shape[1], cfg.patch_size, cfg.stride, "train", cfg.image_size
        )
        val_coords, _, _ = _single_scene_split(
            img.shape[0], img.shape[1], cfg.image_size, cfg.image_size, "validation", cfg.image_size
        )
        test_coords, _, _ = _single_scene_split(
            img.shape[0], img.shape[1], cfg.image_size, cfg.image_size, "test", cfg.image_size
        )
    kwargs = dict(
        img=img, dataset_name=cfg.dataset, scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select, srf_weights=weights, degradation_operator=degradation_operator
    )
    train = HSIHSRDataset(patch_size=cfg.patch_size, coords=train_coords, split="train", augment=True, **kwargs)
    val = HSIHSRDataset(patch_size=cfg.image_size, coords=val_coords, split="validation", augment=False, **kwargs)
    test = HSIHSRDataset(patch_size=cfg.image_size, coords=test_coords, split="test", augment=False, **kwargs)
    return train, val, test, img, weights, names, wavelengths, wavelength_path, profile, val_rect, test_rect


def _build_cave(cfg, degradation_operator):
    root = os.path.join(cfg.data_root, "CAVE")
    if not os.path.isdir(root):
        root = cfg.data_root
    scene_dirs = _find_cave_scene_dirs(root)
    weights, names, wavelengths, wavelength_path, profile, n_select = _make_srf(cfg, 31)
    common = dict(
        scene_dirs=scene_dirs, scale_ratio=cfg.scale_ratio, n_select_bands=n_select,
        srf_weights=weights, degradation_operator=degradation_operator,
    )
    train = CAVEDataset(scene_names=CAVE_TRAIN_SCENES, split="train", patch_size=cfg.patch_size, stride=cfg.stride, augment=True, **common)
    val = CAVEDataset(scene_names=CAVE_VALIDATION_SCENES, split="validation", patch_size=512, stride=512, augment=False, **common)
    test = CAVEDataset(scene_names=CAVE_TEST_SCENES, split="test", patch_size=512, stride=512, augment=False, **common)
    return train, val, test, {
        "dataset":"CAVE", "n_bands":31, "n_select_bands":n_select,
        "train_samples":len(train), "validation_samples":len(val), "test_samples":len(test),
        "validation_rect":None, "test_rect":None, "srf_profile":profile,
        "srf_path":sensor_protocol("CAVE")["srf_path"], "wavelength_path":wavelength_path,
        "srf_weights":weights, "srf_band_names":names, "hsi_wavelengths":wavelengths,
        "protocol":"CAVE deterministic 16 train / 4 validation / 12 test scenes",
    }


def _build_augsburg(cfg, degradation_operator):
    root = _find_augsburg_root(cfg.data_root)
    wavelengths = _read_envi_wavelengths(os.path.join(root, "band_242_meta_info.hdr"))
    weights, names, wavelengths, wavelength_path, profile, n_select = _make_srf(
        cfg, 242, wavelengths
    )
    files = {
        "train": os.path.join(root, "sr_deep_model_data", "EeteS_EnMAP_10m_deep_train.tif"),
        "validation": os.path.join(root, "sr_deep_model_data", "EeteS_EnMAP_10m_deep_valid.tif"),
        "test": os.path.join(root, "sub_area_1", "EeteS_EnMAP_10m_sub_area1.tif"),
    }
    imgs = {k:_read_tiff_cube(v) for k,v in files.items()}
    train_coords = _grid_coords(imgs["train"].shape[0], imgs["train"].shape[1], cfg.patch_size, cfg.stride)
    val_coords = _grid_coords(imgs["validation"].shape[0], imgs["validation"].shape[1], cfg.image_size, cfg.image_size)
    test_coords = _grid_coords(imgs["test"].shape[0], imgs["test"].shape[1], cfg.image_size, cfg.image_size)
    def ds(key, patch, coords, split, aug):
        return HSIHSRDataset(
            imgs[key], "Augsburg", patch, coords, cfg.scale_ratio, n_select,
            split, aug, weights, degradation_operator
        )
    train, val, test = (
        ds("train",cfg.patch_size,train_coords,"train",True),
        ds("validation",cfg.image_size,val_coords,"validation",False),
        ds("test",cfg.image_size,test_coords,"test",False),
    )
    return train, val, test, {
        "dataset":"Augsburg", "n_bands":242, "n_select_bands":n_select,
        "train_samples":len(train), "validation_samples":len(val), "test_samples":len(test),
        "validation_rect":None, "test_rect":None, "srf_profile":profile,
        "srf_path":sensor_protocol("Augsburg")["srf_path"], "wavelength_path":"band_242_meta_info.hdr",
        "srf_weights":weights, "srf_band_names":names, "hsi_wavelengths":wavelengths,
        "protocol":"MDAS official geographic train/validation/sub_area_1 test; synthetic x4",
    }


def build_datasets(cfg, include_validation: bool = False):
    if cfg.dataset not in cfg.datasets:
        raise ValueError(f"Unsupported dataset={cfg.dataset!r}")
    degradation_operator = build_hsi_degradation(cfg)
    print(f"Resolved degradation: {degradation_operator}")

    if cfg.dataset == "CAVE":
        train, val, test, info = _build_cave(cfg, degradation_operator)
    elif cfg.dataset == "Augsburg":
        train, val, test, info = _build_augsburg(cfg, degradation_operator)
    else:
        dcfg = cfg.datasets[cfg.dataset]
        img = read_hsi_mat(os.path.join(cfg.data_root, dcfg.file_name), dcfg.mat_keys)
        (
            train, val, test, img, weights, names, wavelengths, wavelength_path,
            profile, val_rect, test_rect
        ) = _standard_single_scene(cfg, img, degradation_operator)
        info = {
            "dataset":cfg.dataset, "n_bands":int(img.shape[2]),
            "n_select_bands":int(weights.shape[0]) if weights is not None else int(cfg.n_select_bands),
            "train_samples":len(train), "validation_samples":len(val), "test_samples":len(test),
            "validation_rect":val_rect, "test_rect":test_rect, "srf_profile":profile,
            "srf_path":sensor_protocol(cfg.dataset)["srf_path"] if getattr(cfg,"msi_mode","srf")=="srf" else None,
            "wavelength_path":wavelength_path, "srf_weights":weights,
            "srf_band_names":names, "hsi_wavelengths":wavelengths,
            "protocol":(
                "Chikusei centered 2304x2048; top128 test strip + next128 validation strip"
                if cfg.dataset=="Chikusei" else f"{cfg.dataset} center128 test + top-left128 validation"
            ),
        }

    info.update({
        "scale_ratio":cfg.scale_ratio,
        "degradation_mode":degradation_operator.mode,
        "degradation_sigma":getattr(cfg,"degradation_sigma",2.0),
        "degradation_kernel_size":getattr(cfg,"degradation_kernel_size",5),
        "mtf_nyquist":getattr(cfg,"mtf_nyquist",0.2),
        "psf_truncate":getattr(cfg,"psf_truncate",3.0),
        "degradation_repr":repr(degradation_operator),
        "msi_mode":getattr(cfg,"msi_mode","srf"),
    })
    if include_validation:
        return train, val, test, info
    return train, test, info


def build_loaders(cfg):
    train_set, test_set, info = build_datasets(cfg, include_validation=False)
    return (
        _make_loader(train_set,cfg.batch_size,True,cfg.num_workers,True),
        _make_loader(test_set,1,False,0,False),
        info,
    )


def build_train_val_test_loaders(cfg):
    train_set, val_set, test_set, info = build_datasets(cfg, include_validation=True)
    return (
        _make_loader(train_set,cfg.batch_size,True,cfg.num_workers,True),
        _make_loader(val_set,1,False,0,False),
        _make_loader(test_set,1,False,0,False),
        info,
    )
