"""Torch-free CLI configuration for the observation-wise NonRegSRNet protocol."""
from pathlib import Path
import sys

FIT_DEFAULTS = dict(iterations=10000, constant_steps=2000, decay_steps=8000,
                    lr=3e-4, num_rho=100, alpha=1.0, beta=0.01, gamma=0.01,
                    delta=1.0, theta=0.01, ncc_window=1, cache_dir="")


def add_fit_args(parser):
    for name, default in FIT_DEFAULTS.items():
        parser.add_argument("--nonreg_"+name, type=type(default), default=default,
                            help="NonRegSRNet pairwise-fit setting; ignored by pretrained methods")


def append_fit_options(command, options=None):
    for name, default in FIT_DEFAULTS.items():
        flag = "nonreg_"+name
        value = (options or {}).get(flag, default)
        if value != "" and value is not None:
            command += ["--"+flag, str(value)]
    return command


def benchmark_command(root, dataset, mode, device, *, geometry_seeds, cases, eval_seed,
                      data_root, output_override="", max_samples=0, resume_output=False,
                      overwrite_output=False, legacy_single_seed=False, fit_options=None):
    if cases < 1 or max_samples < 0:
        raise ValueError("NonRegSRNet requires cases>=1 and max_samples>=0")
    if mode == "mixed" and not legacy_single_seed:
        if not geometry_seeds or len(set(geometry_seeds)) != len(geometry_seeds) or any(s < 0 for s in geometry_seeds):
            raise ValueError("Invalid NonRegSRNet geometry seeds")
        command = [sys.executable, str(root/"test_multiseed_deformation.py"),
                   "--model","NonRegSRNet","--dataset",dataset,"--mode","mixed",
                   "--device",device,"--data_root",str(data_root),
                   "--geometry_seeds",*map(str,geometry_seeds),"--cases",str(cases),
                   "--eval_seed",str(eval_seed),"--max_samples",str(max_samples)]
        if output_override:
            command += ["--output_dir",str(output_override)]
        if resume_output:
            command.append("--resume_output")
        if overwrite_output:
            command.append("--overwrite_output")
    else:
        output = (Path(output_override)/"result.json" if output_override else
                  root/"comparison"/"NonRegSRNet"/"outputs"/"pairwise"/dataset/(mode+".json"))
        command = [sys.executable,str(root/"comparison"/"NonRegSRNet"/"test_hsi_deformed.py"),
                   "--dataset",dataset,"--device",device,"--data_root",str(data_root),
                   "--test_mode","registered_only" if mode=="registered" else "registered_and_warp",
                   "--cases",str(cases),"--eval_seed",str(eval_seed),"--output_json",str(output)]
    return append_fit_options(command,fit_options)
