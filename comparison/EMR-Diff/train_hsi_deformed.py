"""Stage-2 EMR-Diff training under the UAFL/S2Diff HSI-deformation protocol.

Two-stage use:
  stage 1: Train.py on registered physical-degradation data;
  stage 2: load stage-1 best MODEL WEIGHTS ONLY, reinitialize AdamW, then train
           on registered/deformed LR-HSI observations while HR-MSI and GT-HSI
           stay in the reliable coordinate system.

Formal deformed observation:
    X -> W_phi(X) -> P0 -> Y_H
while HR-MSI = R0(X) remains registered.
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import random
import sys
from pathlib import Path

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
    # Keep EMR-Diff's own model package ahead of UAFL/model.py.
    sys.path.append(str(UAFL_DIR))

from hsi_deformation import (  # noqa: E402
    make_deformed_lr_hsi,
    sample_synthetic_geometry,
    sample_training_geometry_batch,
)
from metrics import MetricAverager, calc_metrics  # noqa: E402
from model.ResShift_model import ResShiftTrainer  # noqa: E402


DATASETS = ["PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"]


def parse_args():
    p = argparse.ArgumentParser(
        description="EMR-Diff stage-2 registered/deformed HSI training"
    )
    p.add_argument("--dataset", default="PaviaU", choices=DATASETS)
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
    p.add_argument(
        "--registered_probability",
        type=float,
        default=0.10,
        help=(
            "Probability that a stage-2 sample uses exact identity P0(X). "
            "Formal unified protocol is 0.10; remaining samples use P0(W_phi(X))."
        ),
    )

    p.add_argument("--epochs", type=int, default=600)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--validation_interval", type=int, default=20)
    p.add_argument("--validation_cases", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=999999)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--eval_seed", type=int, default=1234)

    p.add_argument(
        "--checkpoint_dir",
        default="",
        help="Defaults to comparison/EMR-Diff/checkpoints/hsi_warp_final/<dataset>",
    )
    p.add_argument(
        "--log_dir",
        default="",
        help="Defaults to comparison/EMR-Diff/logs/hsi_warp_final/<dataset>",
    )
    p.add_argument(
        "--init_checkpoint",
        default="",
        help=(
            "Registered stage-1 best checkpoint. Defaults to "
            "comparison/EMR-Diff/checkpoints/physical/<dataset>/best.pth.tar. "
            "Only model weights are loaded; optimizer is reinitialized."
        ),
    )
    p.add_argument(
        "--resume",
        default="",
        help="Resume this stage-2 run including optimizer and epoch.",
    )
    return p.parse_args()


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except TypeError:
        generator = torch.Generator(device=device.type)
    generator.manual_seed(int(seed))
    return generator


def build_trainer(args):
    config_path = THIS_DIR / "config" / "5_step_EMRDiff.yaml"
    configs = OmegaConf.load(config_path)
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
    configs.train.batch = [args.batch_size, 1]
    configs.train.num_workers = args.num_workers
    configs.train.optimizer = "AdamW"
    configs.train.lr = args.lr
    configs.train.weight_decay = args.weight_decay
    configs.train.epochs = args.epochs
    configs.train.eval_seed = args.eval_seed
    configs.train.early_stop_metric = "PSNR"
    configs.train.early_stop_min_delta = args.early_stop_min_delta
    configs.train.early_stop_patience = args.early_stop_patience

    checkpoint_dir = (
        Path(args.checkpoint_dir)
        if args.checkpoint_dir
        else THIS_DIR / "checkpoints" / "hsi_warp_final" / args.dataset
    )
    log_dir = (
        Path(args.log_dir)
        if args.log_dir
        else THIS_DIR / "logs" / "hsi_warp_final" / args.dataset
    )
    output_dir = THIS_DIR / "outputs" / "hsi_warp_final" / args.dataset
    configs.train.checkpoint_dir = str(checkpoint_dir)
    configs.train.log_dir = str(log_dir)
    configs.train.output_dir = str(output_dir)
    return ResShiftTrainer(configs), checkpoint_dir, log_dir


def validate_stage1_checkpoint(path, trainer):
    checkpoint_path = Path(path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Registered stage-1 EMR-Diff checkpoint not found: {checkpoint_path}"
        )
    checkpoint = torch.load(
        checkpoint_path, map_location=trainer.device, weights_only=False
    )
    if checkpoint.get("dataset") != trainer.dataset:
        raise ValueError(
            "Stage-1/2 dataset mismatch: "
            f"checkpoint={checkpoint.get('dataset')} requested={trainer.dataset}. "
            "A two-stage run must use the same dataset in both stages."
        )
    if checkpoint.get("degradation_mode") != "physical":
        raise ValueError("Stage-2 must initialize from stage-1 physical degradation")
    if int(checkpoint.get("state_channels", -1)) != trainer.state_channels:
        raise ValueError(
            "Stage-1 state-channel mismatch: "
            f"checkpoint={checkpoint.get('state_channels')} "
            f"current={trainer.state_channels}."
        )
    state = checkpoint.get("model_state_dict")
    if state is None:
        raise KeyError("Stage-1 checkpoint has no model_state_dict")
    trainer.Net.load_state_dict(state, strict=True)
    print(
        f"Loaded registered stage-1 MODEL WEIGHTS ONLY: {checkpoint_path} "
        f"epoch={checkpoint.get('epoch')} "
        f"best={checkpoint.get('best_score')}"
    )


def save_stage2(
    path, trainer, optimizer, epoch, best_psnr, args, init_checkpoint
):
    payload = {
        "training_stage": "stage2_registered_deformed_mixed",
        "dataset": trainer.dataset,
        "degradation_mode": "physical",
        "state_channels": trainer.state_channels,
        "model_state_dict": trainer.Net.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_warp_psnr": float(best_psnr),
        "init_checkpoint": str(Path(init_checkpoint)),
        "registered_probability": float(args.registered_probability),
        "geometry": {
            "max_translation": float(args.max_translation),
            "max_rotation_deg": float(args.max_rotation_deg),
            "max_local_px": float(args.max_local_px),
            "control_grid": int(args.control_grid),
            "min_jacobian": float(args.min_jacobian),
        },
        "args": vars(args),
    }
    torch.save(payload, path)


def resume_stage2(path, trainer, optimizer):
    checkpoint = torch.load(path, map_location=trainer.device, weights_only=False)
    if checkpoint.get("training_stage") != "stage2_registered_deformed_mixed":
        raise ValueError("Resume checkpoint is not an EMR stage-2 mixed run")
    if checkpoint.get("dataset") != trainer.dataset:
        raise ValueError("Stage-2 resume dataset mismatch")
    if int(checkpoint.get("state_channels", -1)) != trainer.state_channels:
        raise ValueError("Stage-2 resume state-channel mismatch")
    trainer.Net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start = int(checkpoint["epoch"]) + 1
    best = float(checkpoint.get("best_warp_psnr", float("-inf")))
    print(
        f"Resumed EMR stage-2: {path} start_epoch={start} "
        f"best_warp_PSNR={best:.4f}"
    )
    return start, best


def emr_training_loss(trainer, gt, lr_hsi, hr_msi):
    x_start = trainer._x_start(gt)
    condition = trainer._condition(lr_hsi, hr_msi, gt.shape[-2:])
    timesteps = torch.randint(
        0,
        trainer.num_timesteps,
        size=(gt.shape[0],),
        device=trainer.device,
    )
    noise = torch.randn_like(condition)
    x_t = trainer.EMRDIFF.forward_addnoise(
        x_start=x_start,
        y=condition,
        t=timesteps,
        noise=noise,
        rgb_hr=hr_msi,
    )
    lr_hr = F.interpolate(
        lr_hsi,
        size=gt.shape[-2:],
        mode="bicubic",
        align_corners=False,
    )
    network_output, up_out = trainer.Net(
        x_t, hr_msi, lr_hr, timesteps
    )
    return trainer._multiscale_loss(
        network_output, up_out, x_start, lr_hsi, hr_msi
    )


@torch.no_grad()
def evaluate_registered_and_warped(trainer, loader, p0, args):
    trainer.Net.eval()
    reg_meter = MetricAverager()
    warp_meter = MetricAverager()

    cuda_devices = []
    if trainer.device.type == "cuda":
        index = (
            trainer.device.index
            if trainer.device.index is not None
            else torch.cuda.current_device()
        )
        cuda_devices = [index]

    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.eval_seed)

        for batch in loader:
            gt = batch["gt"].to(
                trainer.device, dtype=torch.float32, non_blocking=True
            )
            hr_msi = batch["hr_msi"].to(
                trainer.device, dtype=torch.float32, non_blocking=True
            )

            lr_reg = p0.degrade(gt)
            pred_reg = trainer._reconstruct(lr_reg, hr_msi)
            reg_meter.update(
                calc_metrics(
                    pred_reg.clamp(0, 1), gt, scale_ratio=args.scale_ratio
                )
            )

            val_gen = make_generator(trainer.device, args.seed + 60000)
            for _ in range(int(args.validation_cases)):
                geometry = sample_synthetic_geometry(
                    gt.shape[-2],
                    gt.shape[-1],
                    device=trainer.device,
                    dtype=gt.dtype,
                    generator=val_gen,
                    max_translation=args.max_translation,
                    max_rotation_deg=args.max_rotation_deg,
                    max_local_px=args.max_local_px,
                    control_grid=args.control_grid,
                    min_jacobian=args.min_jacobian,
                )
                lr_warp = make_deformed_lr_hsi(gt, geometry, p0)
                pred_warp = trainer._reconstruct(lr_warp, hr_msi)
                warp_meter.update(
                    calc_metrics(
                        pred_warp.clamp(0, 1),
                        gt,
                        scale_ratio=args.scale_ratio,
                    )
                )

    return reg_meter.average(), warp_meter.average()


HISTORY_FIELDS = [
    "epoch", "train_loss", "registered_fraction", "lr",
    "reg_PSNR", "reg_SSIM", "reg_ERGAS", "reg_SAM", "reg_CC", "reg_RMSE",
    "warp_PSNR", "warp_SSIM", "warp_ERGAS", "warp_SAM", "warp_CC", "warp_RMSE",
]


def append_history(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in HISTORY_FIELDS})


def main():
    args = parse_args()
    if args.scale_ratio != 4:
        raise ValueError("Formal two-stage comparison uses scale_ratio=4")
    if args.degradation_mode != "physical":
        raise ValueError("Formal stage-2 comparison requires physical degradation")
    if args.patch_size != 64 or args.stride != 32 or args.image_size != 128:
        raise ValueError(
            "Formal two-stage protocol requires image_size=128, "
            "patch_size=64, stride=32"
        )
    if not 0.0 <= args.registered_probability <= 1.0:
        raise ValueError("--registered_probability must be in [0,1]")
    if args.validation_cases < 1:
        raise ValueError("--validation_cases must be >=1")
    if args.validation_interval < 1 or args.save_interval < 1:
        raise ValueError("validation/save intervals must be >=1")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    trainer, checkpoint_dir, log_dir = build_trainer(args)
    if trainer.degradation_mode != "physical":
        raise RuntimeError("EMR stage-2 resolved a non-physical degradation")

    train_dataset = trainer.train_dataloader.dataset
    if not hasattr(train_dataset, "degradation_operator"):
        raise AttributeError(
            "Shared training dataset does not expose degradation_operator"
        )
    p0 = train_dataset.degradation_operator.to(trainer.device)

    init_checkpoint = args.init_checkpoint or str(
        THIS_DIR / "checkpoints" / "physical" / args.dataset / "best.pth.tar"
    )

    # Stage 2 always starts a new optimizer unless explicitly resuming stage 2.
    optimizer = torch.optim.AdamW(
        trainer.Net.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    if args.resume:
        start_epoch, best_psnr = resume_stage2(
            args.resume, trainer, optimizer
        )
    else:
        validate_stage1_checkpoint(init_checkpoint, trainer)
        start_epoch, best_psnr = 0, float("-inf")

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "history.csv"

    protocol = (
        "EMR-Diff TWO-STAGE HSI-DEFORMED PROTOCOL\n"
        f"dataset={args.dataset}\n"
        "stage1=registered physical only\n"
        "stage2=registered + deformed LR-HSI mixture\n"
        f"registered_probability={args.registered_probability}\n"
        f"train_patch={args.patch_size}x{args.patch_size}\n"
        f"validation_test_patch={args.image_size}x{args.image_size}\n"
        "deformed_observation=Y_H=P0(W_phi(X)); HR-MSI=R0(X) stays registered\n"
        f"physical_degradation=MTF@Nyquist {args.mtf_nyquist}, "
        f"truncate {args.psf_truncate}, x{args.scale_ratio}\n"
        f"dx_dy=independent U(-{args.max_translation},+{args.max_translation}) HR px\n"
        f"rotation=U(-{args.max_rotation_deg},+{args.max_rotation_deg}) deg\n"
        f"local=max {args.max_local_px} HR px, "
        f"{args.control_grid}x{args.control_grid} controls, cubic B-spline\n"
        f"min_jacobian={args.min_jacobian}\n"
        f"validation_cases={args.validation_cases}, geometry_seed={args.seed + 60000}\n"
        f"diffusion_eval_seed={args.eval_seed}\n"
        "validation_reports=Registered + Warp; best_monitor=Warp PSNR\n"
        f"optimizer=AdamW(lr={args.lr}, wd={args.weight_decay}), "
        f"batch={args.batch_size}\n"
        f"init_checkpoint={init_checkpoint}\n"
        "init_semantics=model weights only; stage2 optimizer reinitialized\n"
    )
    (checkpoint_dir / "run_protocol.txt").write_text(
        protocol, encoding="utf-8"
    )
    print(protocol)

    train_gen = make_generator(trainer.device, args.seed + 404)
    bad_validations = 0

    for epoch in range(start_epoch, args.epochs):
        trainer.Net.train()
        loss_sum = 0.0
        count = 0
        registered_count = 0

        for batch in trainer.train_dataloader:
            gt = batch["gt"].to(
                trainer.device, dtype=torch.float32, non_blocking=True
            )
            hr_msi = batch["hr_msi"].to(
                trainer.device, dtype=torch.float32, non_blocking=True
            )
            batch_size = int(gt.shape[0])

            with torch.no_grad():
                choose_registered = torch.rand(
                    (batch_size,),
                    generator=train_gen,
                    device=trainer.device,
                ) < float(args.registered_probability)

                if bool(choose_registered.all()):
                    lr_hsi = p0.degrade(gt)
                elif bool((~choose_registered).all()):
                    geometry = sample_training_geometry_batch(
                        batch_size,
                        gt.shape[-2],
                        gt.shape[-1],
                        device=trainer.device,
                        dtype=gt.dtype,
                        generator=train_gen,
                        max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg,
                        max_local_px=args.max_local_px,
                        control_grid=args.control_grid,
                        min_jacobian=args.min_jacobian,
                    )
                    lr_hsi = make_deformed_lr_hsi(gt, geometry, p0)
                else:
                    # Batch-safe mixed path: generate both, then select per sample.
                    geometry = sample_training_geometry_batch(
                        batch_size,
                        gt.shape[-2],
                        gt.shape[-1],
                        device=trainer.device,
                        dtype=gt.dtype,
                        generator=train_gen,
                        max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg,
                        max_local_px=args.max_local_px,
                        control_grid=args.control_grid,
                        min_jacobian=args.min_jacobian,
                    )
                    lr_reg = p0.degrade(gt)
                    lr_warp = make_deformed_lr_hsi(gt, geometry, p0)
                    selector = choose_registered.view(-1, 1, 1, 1)
                    lr_hsi = torch.where(selector, lr_reg, lr_warp)

            optimizer.zero_grad(set_to_none=True)
            loss = emr_training_loss(trainer, gt, lr_hsi, hr_msi)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"non-finite EMR stage-2 loss at epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()

            loss_sum += float(loss.item()) * batch_size
            count += batch_size
            registered_count += int(choose_registered.sum().item())

        current = epoch + 1
        row = {
            "epoch": current,
            "train_loss": loss_sum / max(count, 1),
            "registered_fraction": registered_count / max(count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"Epoch {current:04d}/{args.epochs} "
            f"loss={row['train_loss']:.6f} "
            f"registered_fraction={row['registered_fraction']:.3f} "
            f"lr={row['lr']:.3e}"
        )

        do_eval = (
            current % args.validation_interval == 0
            or current == args.epochs
        )
        if do_eval:
            registered, warp = evaluate_registered_and_warped(
                trainer,
                trainer.validation_dataloader,
                p0,
                args,
            )
            print(
                f"  Registered: PSNR={registered['PSNR']:.4f} "
                f"SAM={registered['SAM']:.4f} "
                f"SSIM={registered['SSIM']:.6f} | "
                f"Warp({args.validation_cases} cases): "
                f"PSNR={warp['PSNR']:.4f} SAM={warp['SAM']:.4f} "
                f"SSIM={warp['SSIM']:.6f}"
            )
            for key, value in registered.items():
                row[f"reg_{key}"] = value
            for key, value in warp.items():
                row[f"warp_{key}"] = value
            append_history(history_path, row)

            monitor = float(warp["PSNR"])
            if monitor > best_psnr + args.early_stop_min_delta:
                best_psnr = monitor
                bad_validations = 0
                save_stage2(
                    checkpoint_dir / "best.pth.tar",
                    trainer,
                    optimizer,
                    epoch,
                    best_psnr,
                    args,
                    init_checkpoint,
                )
                print(
                    f"  best updated: warped validation "
                    f"PSNR={best_psnr:.4f}"
                )
            else:
                bad_validations += 1
                print(
                    f"  no improvement >= {args.early_stop_min_delta:.3f} dB "
                    f"({bad_validations}/{args.early_stop_patience})"
                )

            save_stage2(
                checkpoint_dir / "last.pth.tar",
                trainer,
                optimizer,
                epoch,
                best_psnr,
                args,
                init_checkpoint,
            )
            if bad_validations >= args.early_stop_patience:
                print(
                    f"Early stopping at epoch {current}; "
                    f"best warped PSNR={best_psnr:.4f}"
                )
                break

        elif current % args.save_interval == 0:
            save_stage2(
                checkpoint_dir / "last.pth.tar",
                trainer,
                optimizer,
                epoch,
                best_psnr,
                args,
                init_checkpoint,
            )


if __name__ == "__main__":
    main()
