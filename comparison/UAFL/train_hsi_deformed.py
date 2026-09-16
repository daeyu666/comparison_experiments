"""Train UAFL under the S2Diff-MH-matched deformed-HSI acquisition protocol.

Formal observation chain:
    reliable HR-HSI X
      -> synthetic geometry W_phi(X)
      -> fixed physical spatial degradation P0
      -> deformed LR-HSI Y_H

The HR-MSI reference remains registered in the reliable coordinate system.
UAFL itself is unchanged: bicubic-upsampled LR-HSI + HR-MSI -> HR-HSI.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import build_shared_cfg, checkpoint_payload, require_srf_weights, resolve_device, set_seed  # noqa: E402
from data_loader import build_datasets  # noqa: E402
from hsi_deformation import make_deformed_lr_hsi, sample_synthetic_geometry, sample_training_geometry_batch  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from model import PAPER_PARAMETER_M, build_uafl, parameter_count  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="UAFL deformed-HSI acquisition training")
    p.add_argument("--dataset", default="PaviaU", choices=["PaviaU", "Houston13", "Chikusei"])
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--image_size", type=int, default=128, help="validation/test HR patch size")
    p.add_argument("--patch_size", type=int, default=64, help="training HR patch size")
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--scale_ratio", type=int, default=4)
    p.add_argument("--degradation_mode", default="physical", choices=["physical"])
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)
    p.add_argument("--msi_mode", default="srf", choices=["srf"])
    p.add_argument("--srf_path", default="")
    p.add_argument("--wavelength_root", default="./data/wavelengths")
    p.add_argument("--wavelength_path", default="")
    p.add_argument("--srf_interp", default="pchip", choices=["pchip", "linear"])
    p.add_argument("--srf_band_set", default="auto")

    # S2Diff-MH matched geometry defaults.
    p.add_argument("--max_translation", type=float, default=4.0,
                   help="independent dx/dy bound in HR pixels: dx,dy~U(-d,d)")
    p.add_argument("--max_rotation_deg", type=float, default=2.0)
    p.add_argument("--max_local_px", type=float, default=4.0)
    p.add_argument("--control_grid", type=int, default=5)
    p.add_argument("--min_jacobian", type=float, default=0.5)

    # UAFL paper-style optimizer/loss; this run adapts from the registered best.
    p.add_argument("--epochs", type=int, default=400)
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
    p.add_argument("--checkpoint_dir", default="comparison/UAFL/checkpoints/hsi_warp_final/PaviaU")
    p.add_argument("--log_dir", default="comparison/UAFL/logs/hsi_warp_final/PaviaU")
    p.add_argument("--init_checkpoint", default="comparison/UAFL/checkpoints/physical/PaviaU/best.pth.tar",
                   help="registered UAFL checkpoint; model weights only, optimizer is reinitialized")
    p.add_argument("--resume", default="", help="resume this deformed-HSI run including optimizer/epoch")
    return p.parse_args()


def make_loader(dataset, batch_size, shuffle, workers):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def make_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        gen = torch.Generator(device=device)
    except TypeError:
        gen = torch.Generator(device=device.type)
    gen.manual_seed(int(seed))
    return gen


def upsample_lr_hsi(lr_hsi: torch.Tensor, hr_size) -> torch.Tensor:
    return F.interpolate(lr_hsi, size=tuple(hr_size), mode="bicubic", align_corners=False)


def load_model_weights(model, path: str, device: torch.device):
    if not path:
        return
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"UAFL init checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    epoch = ckpt.get("epoch") if isinstance(ckpt, dict) else None
    best = ckpt.get("best_psnr") if isinstance(ckpt, dict) else None
    print(f"Loaded registered UAFL weights only: {ckpt_path} epoch={epoch} best_psnr={best}")


def resume_run(model, optimizer, path: str, device: torch.device):
    if not path:
        return 0, float("-inf")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start = int(ckpt.get("epoch", -1)) + 1
    best = float(ckpt.get("best_psnr", float("-inf")))
    print(f"Resumed deformed-HSI run: {path} start_epoch={start} best_warp_PSNR={best:.4f}")
    return start, best


def synthesize_warped_lr(gt, p0, geometry):
    with torch.no_grad():
        return make_deformed_lr_hsi(gt, geometry, p0)


@torch.no_grad()
def evaluate_registered_and_warped(model, loader, device, p0, args):
    model.eval()
    registered_meter = MetricAverager()
    warp_meter = MetricAverager()

    # Recreate the same validation deformation cases every evaluation.
    val_gen = make_generator(device, args.seed + 60000)
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)

        lr_reg = p0.degrade(gt)
        pred_reg = model(upsample_lr_hsi(lr_reg, gt.shape[-2:]), hr_msi)
        registered_meter.update(calc_metrics(pred_reg.clamp(0, 1), gt, args.scale_ratio))

        for _ in range(int(args.validation_cases)):
            geometry = sample_synthetic_geometry(
                gt.shape[-2],
                gt.shape[-1],
                device=device,
                dtype=gt.dtype,
                generator=val_gen,
                max_translation=args.max_translation,
                max_rotation_deg=args.max_rotation_deg,
                max_local_px=args.max_local_px,
                control_grid=args.control_grid,
                min_jacobian=args.min_jacobian,
            )
            lr_warp = make_deformed_lr_hsi(gt, geometry, p0)
            pred_warp = model(upsample_lr_hsi(lr_warp, gt.shape[-2:]), hr_msi)
            warp_meter.update(calc_metrics(pred_warp.clamp(0, 1), gt, args.scale_ratio))

    return registered_meter.average(), warp_meter.average()


HISTORY_FIELDS = [
    "epoch", "train_l1", "lr",
    "reg_PSNR", "reg_SSIM", "reg_ERGAS", "reg_SAM", "reg_CC", "reg_RMSE",
    "warp_PSNR", "warp_SSIM", "warp_ERGAS", "warp_SAM", "warp_CC", "warp_RMSE",
]


def append_history(path: Path, row: dict):
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=HISTORY_FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in HISTORY_FIELDS})


def main():
    args = parse_args()
    if args.dataset != "PaviaU":
        print("Note: formal requested protocol is PaviaU; other datasets are retained only for code reuse.")
    if args.patch_size % 8 or args.image_size % 8:
        raise ValueError("UAFL SACA requires HR train/validation sizes divisible by 8")
    if args.validation_cases < 1:
        raise ValueError("--validation_cases must be >=1")

    set_seed(args.seed)
    cfg = build_shared_cfg(args)
    train_set, val_set, _test_set, info = build_datasets(cfg, include_validation=True)
    require_srf_weights(info)

    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, 1, False, 0)
    device = resolve_device(args.device)
    p0 = train_set.degradation_operator.to(device)

    model = build_uafl(int(info["n_select_bands"])).to(device)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.resume:
        start_epoch, best_psnr = resume_run(model, optimizer, args.resume, device)
    else:
        load_model_weights(model, args.init_checkpoint, device)
        start_epoch, best_psnr = 0, float("-inf")

    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    history_path = log_dir / "history.csv"

    protocol = (
        "UAFL FORMAL HSI-DEFORMED PROTOCOL\n"
        f"dataset={args.dataset}\n"
        f"train_patch={args.patch_size}x{args.patch_size}\n"
        f"validation_test_patch={args.image_size}x{args.image_size}\n"
        "observation=Y_H=P0(W_phi(X)); HR-MSI=R0(X) stays registered\n"
        f"physical_degradation=MTF@Nyquist {args.mtf_nyquist}, truncate {args.psf_truncate}, x{args.scale_ratio}\n"
        f"dx_dy=independent U(-{args.max_translation},+{args.max_translation}) HR px\n"
        f"rotation=U(-{args.max_rotation_deg},+{args.max_rotation_deg}) deg\n"
        f"local=max {args.max_local_px} HR px, {args.control_grid}x{args.control_grid} controls, cubic B-spline\n"
        f"min_jacobian={args.min_jacobian}\n"
        f"validation_cases={args.validation_cases}, seed={args.seed + 60000}\n"
        "metric_region=full-frame\n"
        f"optimizer=AdamW(lr={args.lr}, wd={args.weight_decay}), batch={args.batch_size}, loss=L1\n"
        f"init_checkpoint={args.init_checkpoint}\n"
        f"model_params={params} ({params/1e6:.4f}M; paper reports ~{PAPER_PARAMETER_M:.2f}M)\n"
    )
    (ckpt_dir / "run_protocol.txt").write_text(protocol, encoding="utf-8")
    print(protocol)

    train_gen = make_generator(device, args.seed + 404)
    bad_validations = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)

            with torch.no_grad():
                geometry = sample_training_geometry_batch(
                    gt.shape[0],
                    gt.shape[-2],
                    gt.shape[-1],
                    device=device,
                    dtype=gt.dtype,
                    generator=train_gen,
                    max_translation=args.max_translation,
                    max_rotation_deg=args.max_rotation_deg,
                    max_local_px=args.max_local_px,
                    control_grid=args.control_grid,
                    min_jacobian=args.min_jacobian,
                )
                lr_hsi = make_deformed_lr_hsi(gt, geometry, p0)
                x_up = upsample_lr_hsi(lr_hsi, gt.shape[-2:])

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_up, hr_msi)
            loss = F.l1_loss(pred, gt)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite UAFL loss at epoch {epoch+1}")
            loss.backward()
            optimizer.step()

            bs = int(gt.shape[0])
            loss_sum += float(loss.item()) * bs
            count += bs

        row = {
            "epoch": epoch + 1,
            "train_l1": loss_sum / max(count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(f"Epoch {epoch+1:04d}/{args.epochs} L1={row['train_l1']:.6f} lr={row['lr']:.3e}")

        do_eval = (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.epochs
        if do_eval:
            reg, warp = evaluate_registered_and_warped(model, val_loader, device, p0, args)
            print(
                f"  Registered: PSNR={reg['PSNR']:.4f} SAM={reg['SAM']:.4f} SSIM={reg['SSIM']:.6f} | "
                f"Warp({args.validation_cases} cases): PSNR={warp['PSNR']:.4f} SAM={warp['SAM']:.4f} SSIM={warp['SSIM']:.6f}"
            )
            for k, v in reg.items():
                row[f"reg_{k}"] = v
            for k, v in warp.items():
                row[f"warp_{k}"] = v
            append_history(history_path, row)

            monitor = float(warp["PSNR"])
            if monitor > best_psnr + args.early_stop_min_delta:
                best_psnr = monitor
                bad_validations = 0
                torch.save(checkpoint_payload(model, optimizer, epoch, best_psnr, args), ckpt_dir / "best.pth.tar")
                print(f"  best updated: warped validation PSNR={best_psnr:.4f}")
            else:
                bad_validations += 1
                print(f"  no improvement >= {args.early_stop_min_delta:.3f} dB ({bad_validations}/{args.early_stop_patience})")

            torch.save(checkpoint_payload(model, optimizer, epoch, best_psnr, args), ckpt_dir / "last.pth.tar")
            if bad_validations >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}; best warped PSNR={best_psnr:.4f}")
                break
        elif (epoch + 1) % args.save_interval == 0:
            torch.save(checkpoint_payload(model, optimizer, epoch, best_psnr, args), ckpt_dir / "last.pth.tar")


if __name__ == "__main__":
    main()
