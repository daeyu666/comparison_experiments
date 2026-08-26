import os
import sys


EMR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_ROOT = os.path.dirname(os.path.dirname(EMR_ROOT))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from config import TrainConfig, get_dataset_configs
from data_loader import build_loaders


def build_ufg_loaders(configs):
    """Build EMR-Diff loaders from comparison_experiments shared data code.

    Current comparison protocol:
      - x4 super-resolution;
      - LR-HSI: 5x5 Gaussian blur (sigma=2) + bicubic downsampling;
      - HR-MSI: 8 uniformly selected bands;
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
    cfg.n_select_bands = int(configs.data.get("msi_bands", 8))

    # comparison_experiments/data_loader.py already implements the agreed
    # Gaussian(5x5, sigma=2) + bicubic LR-HSI degradation.
    cfg.msi_mode = "uniform"

    cfg.batch_size = int(configs.train.get("batch", [1, 1])[0])
    cfg.num_workers = int(configs.train.get("num_workers", 0))
    cfg.device = str(configs.train.get("device", "cuda"))

    train_loader, test_loader, info = build_loaders(cfg)
    info["degradation_mode"] = "gaussian_bicubic"
    info["gaussian_kernel_size"] = int(
        configs.data.get("gaussian_kernel_size", 5)
    )
    info["gaussian_sigma"] = float(configs.data.get("gaussian_sigma", 2.0))
    info["srf_profile"] = None

    cfg.n_select_bands = int(info["n_select_bands"])
    return train_loader, test_loader, info, cfg
