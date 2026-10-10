"""Unified registered / mixed final test for EMR-Diff.

Registered row:
    Y_H = P0(X)
Warp row:
    Y_H = P0(W_phi(X))

The HR-MSI reference and GT-HSI remain in the reliable coordinate system.
The formal unified deformation uses:
    dx,dy ~ U(-4,4)
    rotation ~ U(-2,2)
    local amplitude proposal ~ U(0,4), conditioned on min Jacobian >= 0.5.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
UAFL_DIR = THIS_DIR.parent / "UAFL"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))
if str(UAFL_DIR) not in sys.path:
    sys.path.append(str(UAFL_DIR))

from hsi_deformation import (  # noqa: E402
    jacobian_determinant,
    make_deformed_lr_hsi,
    sample_synthetic_geometry,
)
from model.ResShift_model import ResShiftTrainer  # noqa: E402


DATASETS = ["PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"]


def parse_args():
    p = argparse.ArgumentParser(description="EMR-Diff unified two-stage final test")
    p.add_argument("--dataset", default="PaviaU", choices=DATASETS)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--degradation_mode", default="physical", choices=["physical"])
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--max_translation", type=float, default=4.0)
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)
    p.add_argument("--checkpoint", default="")
    p.add_argument(
        "--test_mode",
        choices=["registered_only", "registered_and_warp"],
        default="registered_and_warp",
    )
    p.add_argument("--print_cases", action="store_true")
    p.add_argument("--output_json", default="")
    return p.parse_args()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))
    return gen


def calc_rmse(pred, target):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    mse = F.mse_loss(pred, target).item()
    return math.sqrt(max(mse, 1e-12))


def calc_psnr(pred, target):
    rmse = calc_rmse(pred, target)
    return 100.0 if rmse <= 1e-12 else 20.0 * math.log10(1.0 / rmse)


def calc_sam(pred, target, eps=1e-12):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    dot = torch.sum(pred * target, dim=1)
    pn = torch.linalg.vector_norm(pred, dim=1)
    tn = torch.linalg.vector_norm(target, dim=1)
    valid = (pn > eps) & (tn > eps)
    if not torch.any(valid):
        return 0.0
    cos = (dot[valid] / (pn[valid] * tn[valid]).clamp_min(eps)).clamp(-1, 1)
    return (torch.acos(cos) * 180.0 / math.pi).mean().item()


def calc_cc(pred, target, eps=1e-8):
    pred = pred.detach().float().view(pred.shape[0], pred.shape[1], -1)
    target = target.detach().float().view(target.shape[0], target.shape[1], -1)
    pc = pred - pred.mean(dim=2, keepdim=True)
    tc = target - target.mean(dim=2, keepdim=True)
    numerator = torch.sum(pc * tc, dim=2)
    denominator = torch.sqrt(
        torch.sum(pc ** 2, dim=2) * torch.sum(tc ** 2, dim=2) + eps
    )
    return torch.mean(numerator / (denominator + eps)).item()


def calc_ergas(pred, target, scale_ratio, eps=1e-8):
    pred = pred.detach().float()
    target = target.detach().float()
    rmse_band = torch.sqrt(torch.mean((pred-target)**2, dim=(0,2,3)) + eps)
    mean_target = torch.mean(target, dim=(0,2,3))
    return (
        100.0 / scale_ratio
        * torch.sqrt(torch.mean((rmse_band/(mean_target+eps))**2))
    ).item()


def calc_ssim_simple(pred, target, eps=1e-8):
    pred = pred.detach().float()
    target = target.detach().float()
    c1, c2 = 0.01**2, 0.03**2
    mx, my = pred.mean(), target.mean()
    vx, vy = pred.var(unbiased=False), target.var(unbiased=False)
    cxy = ((pred-mx)*(target-my)).mean()
    return (
        ((2*mx*my+c1)*(2*cxy+c2))
        / ((mx**2+my**2+c1)*(vx+vy+c2)+eps)
    ).item()


def calc_metrics(pred, target, scale_ratio):
    rmse = calc_rmse(pred, target)
    return {
        "PSNR": calc_psnr(pred, target),
        "SSIM": calc_ssim_simple(pred, target),
        "ERGAS": calc_ergas(pred, target, scale_ratio),
        "SAM": calc_sam(pred, target),
        "CC": calc_cc(pred, target),
        "RMSE": rmse,
        "RMSE_x255": rmse * 255.0,
    }


def average_metrics(rows):
    return {key: float(mean(float(row[key]) for row in rows)) for key in rows[0]}


def format_metrics(m):
    return (
        f"PSNR={m['PSNR']:.4f} SSIM={m['SSIM']:.6f} "
        f"ERGAS={m['ERGAS']:.4f} SAM={m['SAM']:.4f} "
        f"CC={m['CC']:.6f} RMSE={m['RMSE_x255']:.4f} "
        f"(raw={m['RMSE']:.6f})"
    )


def build_trainer(args):
    configs = OmegaConf.load(THIS_DIR / "config" / "5_step_EMRDiff.yaml")
    configs.data.dataset = args.dataset
    configs.data.degradation_mode = "physical"
    configs.data.patch_size = args.patch_size
    configs.data.stride = args.stride
    configs.data.validation_size = args.image_size
    configs.data.test_size = args.image_size
    configs.data.mtf_nyquist = args.mtf_nyquist
    configs.data.psf_truncate = args.psf_truncate
    configs.diffusion.params.sf = args.scale_ratio
    configs.train.device = args.device
    configs.train.seed = args.seed
    configs.train.eval_seed = args.eval_seed
    return ResShiftTrainer(configs)


def main():
    args = parse_args()
    if args.test_mode == "registered_and_warp" and args.cases < 1:
        raise ValueError("--cases must be >=1")
    if (
        args.image_size != 128 or args.patch_size != 64 or args.stride != 32
        or args.scale_ratio != 4
    ):
        raise ValueError("Unified test protocol is fixed at 128/64/32/x4")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer = build_trainer(args)
    test_set = trainer.test_dataloader.dataset
    if len(test_set) < 1:
        raise ValueError("empty EMR-Diff test split")
    if not hasattr(test_set, "degradation_operator"):
        raise AttributeError("test dataset does not expose degradation_operator")
    p0 = test_set.degradation_operator.to(trainer.device)

    default_ckpt = (
        THIS_DIR / "checkpoints"
        / ("physical" if args.test_mode == "registered_only" else "hsi_warp_final")
        / args.dataset / "best.pth.tar"
    )
    ckpt_path = Path(args.checkpoint) if args.checkpoint else default_ckpt
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"EMR-Diff checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(
        ckpt_path, map_location=trainer.device, weights_only=False
    )
    trainer.verify_checkpoint_architecture(
        checkpoint, context=f"final test {ckpt_path}"
    )
    if checkpoint.get("dataset") != args.dataset:
        raise ValueError("checkpoint/test dataset mismatch")
    if int(checkpoint.get("state_channels", -1)) != trainer.state_channels:
        raise ValueError("checkpoint/test state-channel mismatch")
    if checkpoint.get("degradation_mode") != "physical":
        raise ValueError("unified test requires physical-degradation checkpoint")

    if args.test_mode == "registered_only":
        if checkpoint.get("training_stage") not in (None, "stage1_registered"):
            raise ValueError("registered test requires stage-1 checkpoint")
    else:
        if checkpoint.get("training_stage") != "stage2_registered_deformed_mixed":
            raise ValueError("mixed test requires stage-2 EMR checkpoint")
        rp = checkpoint.get("registered_probability")
        if rp is not None and abs(float(rp) - 0.10) > 1e-12:
            print(
                f"WARNING: EMR mixed checkpoint identity probability={rp}, "
                "formal unified value is 0.10."
            )

    trainer.Net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    trainer.Net.eval()

    registered_rows, warp_rows, per_case = [], [], []
    cuda_devices = []
    if trainer.device.type == "cuda":
        idx = (
            trainer.device.index
            if trainer.device.index is not None
            else torch.cuda.current_device()
        )
        cuda_devices = [idx]

    with torch.no_grad(), torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.eval_seed)

        for sample_idx in range(len(test_set)):
            item = test_set[sample_idx]
            gt = item["gt"].unsqueeze(0).to(
                trainer.device, dtype=torch.float32
            )
            hr_msi = item["hr_msi"].unsqueeze(0).to(
                trainer.device, dtype=torch.float32
            )

            registered_lr = p0.degrade(gt)
            registered_pred = trainer._reconstruct(registered_lr, hr_msi)
            registered_rows.append(
                calc_metrics(registered_pred, gt, args.scale_ratio)
            )

            if args.test_mode == "registered_and_warp":
                geometry_gen = make_generator(
                    trainer.device, args.seed + 70000
                )
                for case_idx in range(args.cases):
                    geometry = sample_synthetic_geometry(
                        gt.shape[-2],
                        gt.shape[-1],
                        device=trainer.device,
                        dtype=gt.dtype,
                        generator=geometry_gen,
                        max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg,
                        max_local_px=args.max_local_px,
                        control_grid=args.control_grid,
                        min_jacobian=args.min_jacobian,
                        local_strength_min_fraction=0.0,
                        local_strength_max_fraction=1.0,
                    )
                    warped_lr = make_deformed_lr_hsi(gt, geometry, p0)
                    pred = trainer._reconstruct(warped_lr, hr_msi)
                    metrics = calc_metrics(pred, gt, args.scale_ratio)
                    warp_rows.append(metrics)
                    record = {
                        "sample": sample_idx + 1,
                        "case": case_idx + 1,
                        "dx_hr_px": float(geometry.dx.item()),
                        "dy_hr_px": float(geometry.dy.item()),
                        "rotation_deg": float(geometry.theta_deg.item()),
                        "local_max_hr_px": float(
                            torch.linalg.vector_norm(
                                geometry.local_field, dim=1
                            ).amax().item()
                        ),
                        "min_jacobian": float(
                            jacobian_determinant(
                                geometry.local_field
                            ).amin().item()
                        ),
                        "metrics": metrics,
                    }
                    per_case.append(record)
                    if args.print_cases:
                        print(
                            f"SAMPLE={sample_idx+1:04d} CASE={case_idx+1:02d} "
                            f"dx={record['dx_hr_px']:+.3f} "
                            f"dy={record['dy_hr_px']:+.3f} "
                            f"rot={record['rotation_deg']:+.3f} "
                            f"local={record['local_max_hr_px']:.3f} | "
                            f"{format_metrics(metrics)}"
                        )

    reg = average_metrics(registered_rows)
    warp = average_metrics(warp_rows) if warp_rows else None
    conditions = {
        "method": "EMR-Diff",
        "dataset": args.dataset,
        "test_mode": args.test_mode,
        "checkpoint": str(ckpt_path),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "formal_mixed_identity_probability": 0.10,
        "formal_mixed_deformed_probability": 0.90,
        "warp_dx_dy": "independent U(-4,4) HR pixels",
        "warp_rotation": "U(-2,2) degrees",
        "warp_local_amplitude": "U(0,4) proposal subject to min Jacobian 0.5",
        "cases_per_patch": args.cases if warp is not None else 0,
        "seed": args.seed,
        "eval_seed": args.eval_seed,
    }

    print("=" * 118)
    print("EMR_DIFF_UNIFIED_TWO_STAGE_TEST")
    print(f"REGISTERED {format_metrics(reg)}")
    if warp is not None:
        print(f"WARP       {format_metrics(warp)}")
    print("=" * 118)

    output = {
        "test_conditions": conditions,
        "average_metrics": (
            {"Registered": reg, "Warp": warp}
            if warp is not None else {"Registered": reg}
        ),
        "per_sample_case": per_case,
    }
    if args.output_json:
        out = Path(args.output_json)
    else:
        tag = "registered" if warp is None else "mixed"
        out = (
            THIS_DIR / "outputs" / "unified_two_stage" / args.dataset
            / f"EMRDiff_{tag}_seed{args.seed}.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FINAL_TEST_JSON={out}")


if __name__ == "__main__":
    main()
