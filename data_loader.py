# data_loader.py
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from degradations import build_degradation
from srf_utils import (
    WV2_ALL_8_BANDS,
    WV2_VISIBLE_5_BANDS,
    WV2_VISIBLE_6_BANDS,
    build_srf_weights,
    hsi_to_msi_numpy,
    load_hsi_wavelengths,
    print_srf_summary,
)


IKONOS_4_BANDS = [
    "IKONOS Blue",
    "IKONOS Green",
    "IKONOS Red",
    "IKONOS NIR",
]
WV2_SRF_PATH = "./data/srf/wv2_relative_spectral_response_data_for_i.atcorr.csv"
IKONOS_SRF_PATH = "./data/srf/ikonos_relative_spectral_response.csv"
PAVIA_NOMINAL_WAVELENGTH_PATH = "./data/wavelengths/PaviaU_nominal_430_860.txt"

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


def read_hsi_mat(file_path: str, candidate_keys: List[str]) -> np.ndarray:
    """读取 .mat 高光谱数据并统一返回 H×W×C。"""
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
        for key in candidate_keys:
            if key in mat_data and isinstance(mat_data[key], np.ndarray):
                return fix_hsi_shape(mat_data[key])
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue
            if isinstance(value, np.ndarray) and value.ndim == 3:
                return fix_hsi_shape(value)

    if h5py is not None:
        with h5py.File(file_path, "r") as f:
            for key in candidate_keys:
                if key in f:
                    return fix_hsi_shape(np.array(f[key]))
            for key in f.keys():
                value = np.array(f[key])
                if value.ndim == 3:
                    return fix_hsi_shape(value)

    raise RuntimeError(f"No valid 3D HSI array found in {file_path}")


def fix_hsi_shape(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img).squeeze()
    if img.ndim != 3:
        raise ValueError(f"HSI data must be 3D, but got shape: {img.shape}")

    if img.shape[0] <= 256 and img.shape[1] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (1, 2, 0))
    elif img.shape[1] <= 256 and img.shape[0] > 256 and img.shape[2] > 256:
        img = np.transpose(img, (0, 2, 1))

    return img.astype(np.float32)


def normalize_hsi(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    min_value = float(np.min(img))
    max_value = float(np.max(img))
    if max_value - min_value < 1e-8:
        return np.zeros_like(img, dtype=np.float32)
    return ((img - min_value) / (max_value - min_value)).astype(np.float32)


def crop_to_scale(img: np.ndarray, scale_ratio: int) -> np.ndarray:
    h, w, _ = img.shape
    new_h = h // scale_ratio * scale_ratio
    new_w = w // scale_ratio * scale_ratio
    return img[:new_h, :new_w, :]


def hsi_to_tensor(img: np.ndarray) -> torch.Tensor:
    """H×W×C -> C×H×W。"""
    return torch.from_numpy(img).permute(2, 0, 1).contiguous().float()


def tensor_to_hsi(x: torch.Tensor) -> np.ndarray:
    """C×H×W -> H×W×C。"""
    if x.ndim != 3:
        raise ValueError(f"Expected C×HxW tensor, got {tuple(x.shape)}")
    return x.detach().cpu().permute(1, 2, 0).numpy().astype(np.float32)


def build_hsi_degradation(cfg):
    """构建所有对比方法共享的 LR-HSI 观测退化算子。"""
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
    raise ValueError(
        f"Unsupported degradation_mode={mode!r}; expected gaussian_bicubic or physical"
    )


def make_lr_hsi(
    hr_hsi: np.ndarray,
    scale_ratio: int,
    degradation_operator=None,
) -> np.ndarray:
    """由共享退化算子生成 LR-HSI。"""
    if degradation_operator is None:
        degradation_operator = build_degradation(
            "gaussian_bicubic",
            scale_ratio=scale_ratio,
            sigma=2.0,
            kernel_size=5,
        )

    x = hsi_to_tensor(hr_hsi).unsqueeze(0)
    with torch.no_grad():
        y = degradation_operator.degrade(x).squeeze(0)
    return tensor_to_hsi(y)


def make_hr_msi(hr_hsi: np.ndarray, n_select_bands: int) -> np.ndarray:
    """仅在关闭 SRF 模式时使用的均匀波段回退实现。"""
    n_bands = hr_hsi.shape[2]
    if n_select_bands > n_bands:
        raise ValueError(
            f"n_select_bands={n_select_bands} is larger than HSI bands={n_bands}"
        )
    band_indices = np.linspace(0, n_bands - 1, n_select_bands).round().astype(np.int64)
    return hr_hsi[:, :, band_indices].astype(np.float32)


def get_center_test_rect(h: int, w: int, test_size: int) -> Tuple[int, int, int, int]:
    if h < test_size or w < test_size:
        raise ValueError(
            f"Image size {(h, w)} is smaller than test_size={test_size}."
        )
    top = (h - test_size) // 2
    left = (w - test_size) // 2
    return top, left, top + test_size, left + test_size


def intersects(rect1: Tuple[int, int, int, int], rect2: Tuple[int, int, int, int]) -> bool:
    t1, l1, b1, r1 = rect1
    t2, l2, b2, r2 = rect2
    return not (r1 <= l2 or r2 <= l1 or b1 <= t2 or b2 <= t1)


def get_validation_rect(
    h: int,
    w: int,
    validation_size: int,
    test_rect: Tuple[int, int, int, int],
    dataset_name: str = "",
) -> Tuple[int, int, int, int]:
    """Choose a deterministic validation region disjoint from the center test region.

    PaviaU keeps the established top-left rule so existing PaviaU runs remain
    comparable. Chikusei uses an interior patch centered in the upper-left
    quadrant because a single extreme-corner 128x128 crop can be atypically
    low-energy for this much larger scene.
    """
    if h < validation_size or w < validation_size:
        raise ValueError(
            f"Image size {(h, w)} is smaller than validation_size={validation_size}."
        )

    candidates = []
    if dataset_name == "Chikusei":
        center_y = h // 4
        center_x = w // 4
        top = min(max(center_y - validation_size // 2, 0), h - validation_size)
        left = min(max(center_x - validation_size // 2, 0), w - validation_size)
        candidates.append((top, left, top + validation_size, left + validation_size))

    candidates.extend(
        [
            (0, 0, validation_size, validation_size),
            (0, w - validation_size, validation_size, w),
            (h - validation_size, 0, h, validation_size),
            (h - validation_size, w - validation_size, h, w),
        ]
    )

    for rect in candidates:
        if not intersects(rect, test_rect):
            return rect

    for top in range(0, h - validation_size + 1, validation_size):
        for left in range(0, w - validation_size + 1, validation_size):
            rect = (top, left, top + validation_size, left + validation_size)
            if not intersects(rect, test_rect):
                return rect

    raise ValueError(
        "Cannot place a validation region that is disjoint from the test region."
    )


def build_patch_coords(
    h: int,
    w: int,
    patch_size: int,
    stride: int,
    validation_rect: Tuple[int, int, int, int],
    test_rect: Tuple[int, int, int, int],
    split: str,
) -> List[Tuple[int, int]]:
    split = split.lower()

    if split in ("validation", "val"):
        top, left, bottom, right = validation_rect
        if bottom - top != patch_size or right - left != patch_size:
            raise ValueError(
                f"Validation patch_size={patch_size} must match validation region "
                f"size={(bottom - top, right - left)}."
            )
        return [(top, left)]

    if split == "test":
        top, left, bottom, right = test_rect
        if bottom - top != patch_size or right - left != patch_size:
            raise ValueError(
                f"Test patch_size={patch_size} must match test region "
                f"size={(bottom - top, right - left)}."
            )
        return [(top, left)]

    if split != "train":
        raise ValueError(f"Unsupported split: {split}")

    coords = []
    for top in range(0, h - patch_size + 1, stride):
        for left in range(0, w - patch_size + 1, stride):
            patch_rect = (top, left, top + patch_size, left + patch_size)
            if intersects(patch_rect, validation_rect):
                continue
            if intersects(patch_rect, test_rect):
                continue
            coords.append((top, left))

    if not coords:
        raise RuntimeError(
            "No training patches remain after excluding validation and test regions."
        )
    return coords


class HSIHSRDataset(Dataset):
    """HSI-MSI 融合超分数据集。"""

    def __init__(
        self,
        img: np.ndarray,
        dataset_name: str,
        patch_size: int,
        stride: int,
        scale_ratio: int,
        n_select_bands: int,
        split: str = "train",
        test_size: int = 128,
        validation_size: int = 128,
        augment: bool = True,
        srf_weights=None,
        degradation_operator=None,
    ):
        super().__init__()

        self.img = img
        self.dataset_name = dataset_name
        self.patch_size = patch_size
        self.stride = stride
        self.scale_ratio = scale_ratio
        self.n_select_bands = n_select_bands
        self.split = split
        self.augment = augment and split == "train"
        self.srf_weights = srf_weights
        self.degradation_operator = degradation_operator

        h, w, _ = img.shape
        self.test_rect = get_center_test_rect(h, w, test_size)
        self.validation_rect = get_validation_rect(
            h,
            w,
            validation_size,
            self.test_rect,
            dataset_name=self.dataset_name,
        )
        self.coords = build_patch_coords(
            h=h,
            w=w,
            patch_size=patch_size,
            stride=stride,
            validation_rect=self.validation_rect,
            test_rect=self.test_rect,
            split=split,
        )

    def __len__(self):
        return len(self.coords)

    def random_augment(self, patch: np.ndarray) -> np.ndarray:
        if random.random() < 0.5:
            patch = np.flip(patch, axis=0)
        if random.random() < 0.5:
            patch = np.flip(patch, axis=1)
        if random.random() < 0.5:
            patch = np.rot90(patch, k=random.randint(1, 3), axes=(0, 1))
        return np.ascontiguousarray(patch)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        top, left = self.coords[index]
        gt = self.img[
            top:top + self.patch_size,
            left:left + self.patch_size,
            :,
        ].copy()

        if self.augment:
            gt = self.random_augment(gt)

        lr_hsi = make_lr_hsi(
            gt,
            self.scale_ratio,
            degradation_operator=self.degradation_operator,
        )
        if self.srf_weights is not None:
            hr_msi = hsi_to_msi_numpy(gt, self.srf_weights)
        else:
            hr_msi = make_hr_msi(gt, self.n_select_bands)

        return {
            "lr_hsi": hsi_to_tensor(lr_hsi),
            "hr_msi": hsi_to_tensor(hr_msi),
            "gt": hsi_to_tensor(gt),
            "dataset_id": torch.tensor(0, dtype=torch.long),
            "n_bands": torch.tensor(gt.shape[2], dtype=torch.long),
        }


def _resolve_srf_spec(cfg, n_bands: int):
    """解析固定公平对比传感器及 HSI 波长网格。"""
    requested = getattr(cfg, "srf_band_set", "auto")
    if requested == "auto":
        resolved = "ikonos4" if cfg.dataset == "PaviaU" else "wv2_all8"
    else:
        resolved = requested

    if resolved == "ikonos4":
        selected_bands = IKONOS_4_BANDS
        default_srf_path = IKONOS_SRF_PATH
    elif resolved == "wv2_visible5":
        selected_bands = WV2_VISIBLE_5_BANDS
        default_srf_path = WV2_SRF_PATH
    elif resolved == "wv2_visible6":
        selected_bands = WV2_VISIBLE_6_BANDS
        default_srf_path = WV2_SRF_PATH
    elif resolved == "wv2_all8":
        selected_bands = WV2_ALL_8_BANDS
        default_srf_path = WV2_SRF_PATH
    else:
        raise ValueError(f"Unsupported srf_band_set: {resolved}")

    srf_path = getattr(cfg, "srf_path", "") or default_srf_path

    if getattr(cfg, "wavelength_path", ""):
        wavelength_path = cfg.wavelength_path
        hsi_wavelengths = load_hsi_wavelengths(
            wavelength_path=wavelength_path,
            n_bands=n_bands,
        )
    elif cfg.dataset == "PaviaU" and resolved == "ikonos4":
        wavelength_path = PAVIA_NOMINAL_WAVELENGTH_PATH
        if n_bands == 103 and os.path.exists(wavelength_path):
            hsi_wavelengths = load_hsi_wavelengths(
                wavelength_path=wavelength_path,
                n_bands=n_bands,
            )
        else:
            hsi_wavelengths = np.linspace(430.0, 860.0, n_bands).astype(np.float32)
            wavelength_path = f"nominal:430-860nm/{n_bands}bands"
    else:
        wavelength_path = os.path.join(cfg.wavelength_root, f"{cfg.dataset}.txt")
        hsi_wavelengths = load_hsi_wavelengths(
            wavelength_path=wavelength_path,
            n_bands=n_bands,
        )

    cfg.resolved_srf_band_set = resolved
    cfg.resolved_srf_path = srf_path
    cfg.resolved_wavelength_path = wavelength_path
    return srf_path, selected_bands, hsi_wavelengths, wavelength_path, resolved


def build_datasets(cfg, include_validation: bool = False):
    dataset_cfg = cfg.datasets[cfg.dataset]
    file_path = os.path.join(cfg.data_root, dataset_cfg.file_name)

    img = read_hsi_mat(file_path, dataset_cfg.mat_keys)
    img = normalize_hsi(img)
    img = crop_to_scale(img, cfg.scale_ratio)

    n_bands = img.shape[2]
    print(f"Loaded {cfg.dataset}: shape={img.shape}, bands={n_bands}")

    degradation_operator = build_hsi_degradation(cfg)
    print(f"Resolved degradation: {degradation_operator}")

    srf_weights = None
    srf_band_names = None
    hsi_wavelengths = None
    resolved_band_set = None
    resolved_srf_path = None
    resolved_wavelength_path = None

    if getattr(cfg, "msi_mode", "srf") == "srf":
        (
            resolved_srf_path,
            selected_bands,
            hsi_wavelengths,
            resolved_wavelength_path,
            resolved_band_set,
        ) = _resolve_srf_spec(cfg, n_bands)

        srf_weights, srf_band_names = build_srf_weights(
            srf_path=resolved_srf_path,
            hsi_wavelengths=hsi_wavelengths,
            selected_bands=selected_bands,
            interp_kind=cfg.srf_interp,
            normalize=True,
        )

        print(
            f"Resolved SRF: dataset={cfg.dataset}, profile={resolved_band_set}, "
            f"path={resolved_srf_path}, wavelength_grid={resolved_wavelength_path}"
        )
        print_srf_summary(
            srf_weights=srf_weights,
            band_names=srf_band_names,
            hsi_wavelengths=hsi_wavelengths,
        )
        n_select_bands = srf_weights.shape[0]
    else:
        n_select_bands = cfg.n_select_bands

    validation_size = int(getattr(cfg, "validation_size", cfg.image_size))
    test_size = int(cfg.image_size)
    test_rect = get_center_test_rect(img.shape[0], img.shape[1], test_size)
    validation_rect = get_validation_rect(
        img.shape[0],
        img.shape[1],
        validation_size,
        test_rect,
        dataset_name=cfg.dataset,
    )
    print(
        f"Spatial split: validation_rect={validation_rect}, test_rect={test_rect}; "
        "training patches exclude both regions."
    )

    vt, vl, vb, vr = validation_rect
    validation_gt = img[vt:vb, vl:vr, :]
    print(
        "Validation GT stats: "
        f"min={float(validation_gt.min()):.6f}, "
        f"mean={float(validation_gt.mean()):.6f}, "
        f"max={float(validation_gt.max()):.6f}, "
        f"std={float(validation_gt.std()):.6f}"
    )

    dataset_kwargs = dict(
        img=img,
        dataset_name=cfg.dataset,
        scale_ratio=cfg.scale_ratio,
        n_select_bands=n_select_bands,
        srf_weights=srf_weights,
        degradation_operator=degradation_operator,
        test_size=test_size,
        validation_size=validation_size,
    )

    train_set = HSIHSRDataset(
        patch_size=cfg.patch_size,
        stride=cfg.stride,
        split="train",
        augment=True,
        **dataset_kwargs,
    )
    validation_set = HSIHSRDataset(
        patch_size=validation_size,
        stride=validation_size,
        split="validation",
        augment=False,
        **dataset_kwargs,
    )
    test_set = HSIHSRDataset(
        patch_size=test_size,
        stride=test_size,
        split="test",
        augment=False,
        **dataset_kwargs,
    )

    info = {
        "dataset": cfg.dataset,
        "n_bands": n_bands,
        "n_select_bands": n_select_bands,
        "scale_ratio": cfg.scale_ratio,
        "train_samples": len(train_set),
        "validation_samples": len(validation_set),
        "test_samples": len(test_set),
        "validation_rect": validation_rect,
        "validation_gt_min": float(validation_gt.min()),
        "validation_gt_mean": float(validation_gt.mean()),
        "validation_gt_max": float(validation_gt.max()),
        "validation_gt_std": float(validation_gt.std()),
        "test_rect": test_rect,
        "degradation_mode": degradation_operator.mode,
        "degradation_sigma": getattr(cfg, "degradation_sigma", 2.0),
        "degradation_kernel_size": getattr(cfg, "degradation_kernel_size", 5),
        "mtf_nyquist": getattr(cfg, "mtf_nyquist", 0.2),
        "psf_truncate": getattr(cfg, "psf_truncate", 3.0),
        "degradation_repr": repr(degradation_operator),
        "msi_mode": getattr(cfg, "msi_mode", "srf"),
        "srf_profile": resolved_band_set,
        "srf_path": resolved_srf_path,
        "wavelength_path": resolved_wavelength_path,
        "srf_weights": srf_weights,
        "srf_band_names": srf_band_names,
        "hsi_wavelengths": hsi_wavelengths,
    }

    if include_validation:
        return train_set, validation_set, test_set, info
    return train_set, test_set, info


def _make_loader(dataset, batch_size, shuffle, num_workers, drop_last):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


def build_loaders(cfg):
    """Backward-compatible train/test loader API.

    Training patches still exclude the fixed validation region, even when a caller
    does not request a validation loader.
    """
    train_set, test_set, info = build_datasets(cfg, include_validation=False)
    train_loader = _make_loader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    test_loader = _make_loader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, test_loader, info


def build_train_val_test_loaders(cfg):
    """Return leakage-free train/validation/test loaders for early stopping."""
    train_set, validation_set, test_set, info = build_datasets(
        cfg, include_validation=True
    )
    train_loader = _make_loader(
        train_set,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        drop_last=True,
    )
    validation_loader = _make_loader(
        validation_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    test_loader = _make_loader(
        test_set,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    return train_loader, validation_loader, test_loader, info
