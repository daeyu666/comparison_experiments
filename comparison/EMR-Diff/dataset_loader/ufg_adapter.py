import os
import sys


EMR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(EMR_ROOT))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import TrainConfig, get_dataset_configs
from data_loader import build_train_val_test_loaders


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
    """Build leakage-free EMR-Diff loaders from the shared protocol."""
    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()

    requested_dataset = str(configs.data.get("dataset", "PaviaU"))
    if requested_dataset not in cfg.datasets:
        raise ValueError(f"Unsupported comparison dataset: {requested_dataset}")
    cfg.dataset = requested_dataset

    cfg.data_root = str(
        configs.data.get("data_root", os.path.join(REPO_ROOT, "data", "raw"))
    )
    if not os.path.isabs(cfg.data_root):
        cfg.data_root = os.path.abspath(os.path.join(EMR_ROOT, cfg.data_root))

    dataset_cfg = cfg.datasets[cfg.dataset]
    data_file = os.path.abspath(os.path.join(cfg.data_root, dataset_cfg.file_name))
    print(f"[adapter] dataset={cfg.dataset}, data_file={data_file}")

    cfg.image_size = int(configs.data.get("test_size", 128))
    cfg.validation_size = int(configs.data.get("validation_size", 128))
    cfg.patch_size = int(configs.data.get("patch_size", 64))
    cfg.stride = int(configs.data.get("stride", 32))
    cfg.scale_ratio = int(configs.diffusion.params.get("sf", 4))

    cfg.degradation_mode = str(
        configs.data.get("degradation_mode", "gaussian_bicubic")
    )
    cfg.degradation_sigma = float(configs.data.get("degradation_sigma", 2.0))
    cfg.degradation_kernel_size = int(
        configs.data.get("degradation_kernel_size", 5)
    )
    cfg.mtf_nyquist = float(configs.data.get("mtf_nyquist", 0.2))
    cfg.psf_truncate = float(configs.data.get("psf_truncate", 3.0))

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

    train_loader, validation_loader, test_loader, info = (
        build_train_val_test_loaders(cfg)
    )

    resolved_dataset = str(info.get("dataset"))
    if resolved_dataset != cfg.dataset:
        raise RuntimeError(
            "Dataset protocol mismatch inside adapter: "
            f"requested={cfg.dataset}, resolved={resolved_dataset}."
        )

    expected_channels = EXPECTED_MSI_CHANNELS[cfg.dataset]
    actual_channels = int(info["n_select_bands"])
    if actual_channels != expected_channels:
        raise ValueError(
            f"{cfg.dataset} comparison protocol requires {expected_channels} MSI "
            f"channels, but shared loader produced {actual_channels}."
        )

    cfg.n_select_bands = actual_channels
    return train_loader, validation_loader, test_loader, info, cfg
