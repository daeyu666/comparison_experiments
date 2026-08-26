import os
import sys


EMR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(EMR_ROOT))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import TrainConfig, get_dataset_configs
from data_loader import build_loaders


EXPECTED_MSI_CHANNELS = {
    "PaviaU": 4,
    "Houston13": 8,
    "Chikusei": 8,
}


def _resolve_sensor_paths(dataset):
    """Resolve the fixed sensor protocol used by every comparison experiment."""
    if dataset == "PaviaU":
        srf_band_set = "ikonos4"
        srf_path = os.path.join(
            REPO_ROOT, "data", "srf", "ikonos_relative_spectral_response.csv"
        )
        wavelength_path = os.path.join(
            REPO_ROOT, "data", "wavelengths", "PaviaU_nominal_430_860.txt"
        )
    elif dataset in ("Houston13", "Chikusei"):
        srf_band_set = "wv2_all8"
        srf_path = os.path.join(
            REPO_ROOT,
            "data",
            "srf",
            "wv2_relative_spectral_response_data_for_i.atcorr.csv",
        )
        wavelength_path = os.path.join(
            REPO_ROOT, "data", "wavelengths", f"{dataset}.txt"
        )
    else:
        raise ValueError(f"Unsupported comparison dataset: {dataset}")

    return srf_band_set, srf_path, wavelength_path


def build_ufg_loaders(configs):
    """Build EMR-Diff loaders from the shared comparison protocol.

    Fixed protocol for all comparison methods:
      - x4 super-resolution;
      - LR-HSI: 5x5 Gaussian blur (sigma=2) + bicubic downsampling;
      - PaviaU HR-MSI: IKONOS Blue/Green/Red/NIR, 4 channels;
      - Houston13 / Chikusei HR-MSI: WorldView-2 all8, 8 channels;
      - 64x64 train patches, stride 32;
      - center 128x128 test region.
    """
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()

    cfg.dataset = str(configs.data.get("dataset", "PaviaU"))
    cfg.data_root = str(
        configs.data.get("data_root", os.path.join(REPO_ROOT, "data", "raw"))
    )
    if not os.path.isabs(cfg.data_root):
        cfg.data_root = os.path.abspath(os.path.join(EMR_ROOT, cfg.data_root))

    cfg.image_size = int(configs.data.get("test_size", 128))
    cfg.patch_size = int(configs.data.get("patch_size", 64))
    cfg.stride = int(configs.data.get("stride", 32))
    cfg.scale_ratio = int(configs.diffusion.params.get("sf", 4))

    # Spatial degradation is shared by every comparison method.
    # comparison_experiments/data_loader.py implements Gaussian(5x5, sigma=2)
    # followed by bicubic downsampling.
    cfg.msi_mode = "srf"
    (
        cfg.srf_band_set,
        cfg.srf_path,
        cfg.wavelength_path,
    ) = _resolve_sensor_paths(cfg.dataset)
    cfg.wavelength_root = os.path.join(REPO_ROOT, "data", "wavelengths")

    cfg.batch_size = int(configs.train.get("batch", [1, 1])[0])
    cfg.num_workers = int(configs.train.get("num_workers", 0))
    cfg.device = str(configs.train.get("device", "cuda"))

    train_loader, test_loader, info = build_loaders(cfg)
    info["degradation_mode"] = "gaussian_bicubic"
    info["gaussian_kernel_size"] = int(
        configs.data.get("gaussian_kernel_size", 5)
    )
    info["gaussian_sigma"] = float(configs.data.get("gaussian_sigma", 2.0))

    expected_channels = EXPECTED_MSI_CHANNELS[cfg.dataset]
    actual_channels = int(info["n_select_bands"])
    if actual_channels != expected_channels:
        raise ValueError(
            f"{cfg.dataset} comparison protocol requires {expected_channels} MSI "
            f"channels, but shared loader produced {actual_channels}."
        )

    cfg.n_select_bands = actual_channels
    return train_loader, test_loader, info, cfg
