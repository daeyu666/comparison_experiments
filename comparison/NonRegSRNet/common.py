"""Shared sensor/data and geometry protocol for observation-wise evaluation."""
from dataclasses import fields
from pathlib import Path
import torch

from comparison.UAFL.common import build_shared_cfg, require_srf_weights, set_seed
from comparison.UAFL.hsi_deformation import make_deformed_lr_hsi, sample_synthetic_geometry
from experiment_protocol import DATASETS
from .model import FitConfig
from .options import add_fit_args

HERE = Path(__file__).resolve().parent


def add_data_args(p):
    p.add_argument("--dataset",choices=DATASETS,default="PaviaU")
    p.add_argument("--data_root",default="./data/raw")
    for name,value in dict(image_size=128,patch_size=64,stride=32,scale_ratio=4).items():
        p.add_argument("--"+name,type=int,default=value)
    p.add_argument("--degradation_mode",choices=["physical","gaussian_bicubic"],default="physical")
    p.add_argument("--mtf_nyquist",type=float,default=0.2)
    p.add_argument("--psf_truncate",type=float,default=3.0)
    p.add_argument("--degradation_sigma",type=float,default=2.0)
    p.add_argument("--degradation_kernel_size",type=int,default=5)
    p.add_argument("--msi_mode",choices=["srf"],default="srf")
    p.add_argument("--srf_path",default="")
    p.add_argument("--wavelength_root",default="./data/wavelengths")
    p.add_argument("--wavelength_path",default="")
    p.add_argument("--srf_interp",choices=["pchip","linear"],default="pchip")
    p.add_argument("--srf_band_set",default="auto")
    p.add_argument("--device",default="cuda:0")


def add_geometry_args(p):
    for name,value in dict(max_translation=4.0,max_rotation_deg=2.0,max_local_px=4.0,
                           control_grid=5,min_jacobian=0.5).items():
        p.add_argument("--"+name,type=type(value),default=value)


def fit_config(args):
    default=FitConfig()
    values={item.name:getattr(args,"nonreg_"+item.name,getattr(default,item.name))
            for item in fields(default) if item.name!="seed"}
    values["seed"]=int(getattr(args,"eval_seed",1234))
    return FitConfig(**values).validate()


def resolve_device(name):
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu for a CPU smoke test")
    return torch.device(name)


def geometry_kwargs(args):
    return {name:getattr(args,name) for name in
            ("max_translation","max_rotation_deg","max_local_px","control_grid","min_jacobian")}
