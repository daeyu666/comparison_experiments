# config.py
import argparse
import os
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DatasetConfig:
    """单个数据集的配置信息。"""
    name: str
    file_name: str
    mat_keys: list
    n_select_bands: int = 5


@dataclass
class TrainConfig:
    """通用训练配置，不依赖具体模型结构。"""

    # --- 路径 ---
    project_root: str = "."
    data_root: str = "./data/raw"
    cache_root: str = "./data/cache"
    checkpoint_root: str = "./checkpoints"
    log_root: str = "./logs"
    output_root: str = "./outputs"

    # --- 运行阶段 ---
    stage: str = "train"
    dataset: str = "PaviaU"

    # --- 数据 ---
    image_size: int = 128
    patch_size: int = 64
    stride: int = 32
    scale_ratio: int = 4
    n_select_bands: int = 4

    # --- LR-HSI 退化模式 ---
    # gaussian_bicubic: 5x5 Gaussian(sigma=2) + bicubic x4
    # physical: MTF@Nyquist -> Gaussian optical PSF -> detector area integration -> sampling
    degradation_mode: str = "gaussian_bicubic"
    degradation_sigma: float = 2.0
    degradation_kernel_size: int = 5
    mtf_nyquist: float = 0.2
    psf_truncate: float = 3.0

    # --- MSI 生成模式 ---
    # 公平对比固定使用真实 SRF：PaviaU -> IKONOS 4-band；
    # Houston13 / Chikusei -> WV2 all8。
    msi_mode: str = "srf"
    srf_path: str = ""
    wavelength_root: str = "./data/wavelengths"
    wavelength_path: str = ""
    srf_interp: str = "pchip"
    srf_band_set: str = "auto"

    # --- 训练 ---
    epochs: int = 300
    batch_size: int = 4
    num_workers: int = 0
    lr: float = 1e-4
    weight_decay: float = 0.0
    seed: int = 10
    device: str = "cuda"

    # --- 损失权重 ---
    lambda_l1: float = 1.0
    lambda_sam: float = 0.1
    lambda_dc: float = 0.1
    lambda_sgrad: float = 0.05
    lambda_sdir: float = 0.2
    lambda_ns_l1: float = 1.0
    lambda_srf_region: float = 0.3
    lambda_mse: float = 1.0

    # --- 保存 / 恢复 ---
    save_interval: int = 20
    eval_interval: int = 1
    resume: str = ""
    save_name: str = ""

    datasets: dict = field(default_factory=dict)


def get_dataset_configs():
    return {
        "PaviaU": DatasetConfig(
            name="PaviaU",
            file_name="PaviaU.mat",
            mat_keys=["paviaU", "PaviaU", "img", "data"],
            n_select_bands=4,
        ),
        "Houston13": DatasetConfig(
            name="Houston13",
            file_name="Houston13.mat",
            mat_keys=["Houston13", "Houston_HSI", "data", "img"],
            n_select_bands=8,
        ),
        "Chikusei": DatasetConfig(
            name="Chikusei",
            file_name="Chikusei.mat",
            mat_keys=["chikusei", "Chikusei", "img", "data"],
            n_select_bands=8,
        ),
    }


def parse_args(argv: Optional[List[str]] = None):
    parser = argparse.ArgumentParser(description="HSI Super-Resolution Template")

    parser.add_argument("--stage", type=str, default="train")
    parser.add_argument("--dataset", type=str, default="PaviaU")

    parser.add_argument("--data_root", type=str, default="./data/raw")
    parser.add_argument("--checkpoint_root", type=str, default="./checkpoints")
    parser.add_argument("--log_root", type=str, default="./logs")
    parser.add_argument("--output_root", type=str, default="./outputs")

    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--patch_size", type=int, default=64)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--scale_ratio", type=int, default=4)
    parser.add_argument(
        "--n_select_bands",
        type=int,
        default=0,
        help="0 表示使用数据集默认通道数；SRF 模式下由实际 SRF 通道数覆盖。",
    )

    parser.add_argument(
        "--degradation_mode",
        type=str,
        default="gaussian_bicubic",
        choices=["gaussian_bicubic", "physical"],
        help="所有对比实验统一支持常规退化与物理退化切换。",
    )
    parser.add_argument("--degradation_sigma", type=float, default=2.0)
    parser.add_argument("--degradation_kernel_size", type=int, default=5)
    parser.add_argument("--mtf_nyquist", type=float, default=0.2)
    parser.add_argument("--psf_truncate", type=float, default=3.0)

    parser.add_argument(
        "--msi_mode",
        type=str,
        default="srf",
        choices=["uniform", "srf"],
    )
    parser.add_argument(
        "--srf_path",
        type=str,
        default="",
        help="可选的显式 SRF CSV；留空时按 srf_band_set 自动选择。",
    )
    parser.add_argument("--wavelength_root", type=str, default="./data/wavelengths")
    parser.add_argument("--wavelength_path", type=str, default="")
    parser.add_argument(
        "--srf_interp",
        type=str,
        default="pchip",
        choices=["pchip", "linear"],
    )
    parser.add_argument(
        "--srf_band_set",
        type=str,
        default="auto",
        choices=["auto", "ikonos4", "wv2_visible5", "wv2_visible6", "wv2_all8"],
        help="auto: PaviaU 使用 IKONOS4；Houston13/Chikusei 使用 WV2 all8。",
    )

    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--lambda_l1", type=float, default=1.0)
    parser.add_argument("--lambda_sam", type=float, default=0.1)
    parser.add_argument("--lambda_dc", type=float, default=0.1)
    parser.add_argument("--lambda_sgrad", type=float, default=0.05)
    parser.add_argument("--lambda_sdir", type=float, default=0.2)
    parser.add_argument("--lambda_ns_l1", type=float, default=1.0)
    parser.add_argument("--lambda_srf_region", type=float, default=0.3)
    parser.add_argument("--lambda_mse", type=float, default=1.0)

    parser.add_argument("--save_interval", type=int, default=20)
    parser.add_argument("--eval_interval", type=int, default=1)
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--save_name", type=str, default="")

    args = parser.parse_args(argv)

    cfg = TrainConfig()
    cfg.datasets = get_dataset_configs()

    for key, value in vars(args).items():
        setattr(cfg, key, value)

    dataset_cfg = cfg.datasets.get(cfg.dataset)
    if dataset_cfg is not None:
        cfg.n_select_bands = args.n_select_bands or dataset_cfg.n_select_bands

    make_dirs(cfg)
    return cfg


def make_dirs(cfg: TrainConfig):
    dirs = [
        cfg.checkpoint_root,
        cfg.log_root,
        cfg.output_root,
        os.path.join(cfg.output_root, "predictions", cfg.dataset),
        os.path.join(cfg.output_root, "metrics"),
        os.path.join(cfg.output_root, "figures"),
    ]
    for path in dirs:
        os.makedirs(path, exist_ok=True)


def get_checkpoint_path(cfg: TrainConfig, stage: str = None, name: str = None):
    stage = stage or cfg.stage
    if name is None or name == "":
        name = f"{cfg.dataset}_{stage}.pth"
    return os.path.join(cfg.checkpoint_root, stage, name)


def print_config(cfg: TrainConfig):
    print("=" * 60)
    print("HSI Super-Resolution Template  Config")
    print("=" * 60)
    for key, value in cfg.__dict__.items():
        if key != "datasets":
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    cfg = parse_args()
    print_config(cfg)
