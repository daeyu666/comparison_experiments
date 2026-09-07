"""Train HSIFN on comparison_experiments shared HSI-MSI protocol."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from common import (  # noqa: E402
    MISALIGNMENT_MODES,
    build_shared_cfg,
    calc_masked_psnr_sam,
    checkpoint_payload,
    require_srf_weights,
    resolve_device,
    set_seed,
    warp_msi_for_protocol,
)
from data_loader import build_datasets  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from model import HSIFN  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="HSIFN unregistered HSI-MSI fusion")
    p.add_argument("--dataset", default="PaviaU", choices=["PaviaU", "Houston13", "Chikusei"])
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--image_size", type=int, default=128)
    p.add_argument("--patch_size", type=int, default=64)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--scale_ratio", type=int, default=4)

    p.add_argument("--degradation_mode", default="physical", choices=["physical", "gaussian_bicubic"])
    p.add_argument("--degradation_sigma", type=float, default=2.0)
    p.add_argument("--degradation_kernel_size", type=int, default=5)
    p.add_argument("--mtf_nyquist", type=float, default=0.2)
    p.add_argument("--psf_truncate", type=float, default=3.0)

    p.add_argument("--msi_mode", default="srf", choices=["srf", "uniform"])
    p.add_argument("--srf_path", default="")
    p.add_argument("--wavelength_root", default="./data/wavelengths")
    p.add_argument("--wavelength_path", default="")
    p.add_argument("--srf_interp", default="pchip", choices=["pchip", "linear"])
    p.add_argument(
        "--srf_band_set",
        default="auto",
        choices=["auto", "ikonos4", "wv2_visible5", "wv2_visible6", "wv2_all8"],
    )

    p.add_argument("--train_misalignment_mode", default="registered", choices=MISALIGNMENT_MODES)
    p.add_argument("--translation_max_px", type=float, default=0.0)
    p.add_argument("--rotation_max_deg", type=float, default=0.0)
    p.add_argument("--local_max_displacement_px", type=float, default=0.0)
    p.add_argument("--control_grid_size", type=int, default=5)
    p.add_argument("--valid_threshold", type=float, default=0.999)

    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    # Official real-unaligned HSIFN configuration uses AdamW lr=1e-5.
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--grad_clip", type=float, default=1e6)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--no_mask", action="store_true")

    p.add_argument("--validation_interval", type=int, default=0,
                   help="0 uses PaviaU=20, Houston13=10, Chikusei=5")
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--resume", default="")
    return p.parse_args()


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def load_resume(model, optimizer, path: str, device: torch.device):
    if not path:
        return 0, float("-inf")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0
    best = float(ckpt.get("best_psnr", float("-inf"))) if isinstance(ckpt, dict) else float("-inf")
    print(f"Resumed {path}: start_epoch={start}, best_valid_PSNR={best:.4f}")
    return start, best


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    warp_gen = torch.Generator(device="cpu").manual_seed(args.eval_seed + 100003)
    rows = []
    flow_means = []
    mask_means = []
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        warped_msi, valid, _ = warp_msi_for_protocol(
            hr_msi,
            mode=args.train_misalignment_mode,
            translation_max_px=args.translation_max_px,
            rotation_max_deg=args.rotation_max_deg,
            local_max_displacement_px=args.local_max_displacement_px,
            control_grid_size=args.control_grid_size,
            generator=warp_gen,
        )
        pred, diag = model(lr_hsi, warped_msi)
        pred_metric = pred.clamp(0.0, 1.0)
        full = calc_metrics(pred_metric, gt, args.scale_ratio)
        valid_psnr, valid_sam, valid_fraction = calc_masked_psnr_sam(
            pred_metric, gt, valid, threshold=args.valid_threshold
        )
        rows.append((full, valid_psnr, valid_sam, valid_fraction))
        flow_means.append(
            torch.linalg.vector_norm(diag["coarse_flow"], dim=1).mean().item()
        )
        if diag["masks"]:
            mask_means.append(torch.stack([m.mean() for m in diag["masks"]]).mean().item())

    mean_full = {
        key: sum(row[0][key] for row in rows) / len(rows)
        for key in rows[0][0]
    }
    return {
        **mean_full,
        "PSNR_valid": sum(r[1] for r in rows) / len(rows),
        "SAM_valid": sum(r[2] for r in rows) / len(rows),
        "valid_fraction": sum(r[3] for r in rows) / len(rows),
        "flow_mean_px": sum(flow_means) / len(flow_means),
        "mask_mean": sum(mask_means) / len(mask_means) if mask_means else float("nan"),
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    cfg = build_shared_cfg(args)
    train_set, val_set, _, info = build_datasets(cfg, include_validation=True)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, 1, False, 0)

    device = resolve_device(args.device)
    srf = require_srf_weights(info)
    model = HSIFN(
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        srf_weights=srf,
        use_mask=not args.no_mask,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")

    if not args.checkpoint_dir:
        args.checkpoint_dir = str(THIS_DIR / "checkpoints" / args.degradation_mode / args.dataset)
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    validation_interval = args.validation_interval or {
        "PaviaU": 20, "Houston13": 10, "Chikusei": 5
    }[args.dataset]

    with open(os.path.join(args.checkpoint_dir, "run_protocol.txt"), "w", encoding="utf-8") as f:
        f.write(f"method: HSIFN\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"degradation_mode: {args.degradation_mode}\n")
        f.write(f"srf_profile: {info.get('srf_profile')}\n")
        f.write(f"hsi_bands: {info['n_bands']}\n")
        f.write(f"msi_bands: {info['n_select_bands']}\n")
        f.write(f"train_patch: {args.patch_size}\n")
        f.write(f"train_stride: {args.stride}\n")
        f.write(f"validation_size: {args.image_size}\n")
        f.write(f"validation_interval: {validation_interval}\n")
        f.write("reference_frame: fixed HSI/GT; warp HR-MSI only\n")
        f.write(f"train_misalignment_mode: {args.train_misalignment_mode}\n")
        f.write(f"translation_max_px: {args.translation_max_px}\n")
        f.write(f"rotation_max_deg: {args.rotation_max_deg}\n")
        f.write(f"local_max_displacement_px: {args.local_max_displacement_px}\n")
        f.write(f"control_grid_size: {args.control_grid_size}\n")
        f.write("early_stop_metric: PSNR_valid\n")
        f.write(f"early_stop_min_delta: {args.early_stop_min_delta}\n")
        f.write(f"early_stop_patience: {args.early_stop_patience}\n")
        f.write(f"eval_seed: {args.eval_seed}\n")

    start_epoch, best_psnr = load_resume(model, optimizer, args.resume, device)
    bad_validations = 0
    train_warp_gen = torch.Generator(device="cpu").manual_seed(args.seed + 7919)

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"HSIFN: dataset={args.dataset}, HSI={info['n_bands']}, "
        f"MSI={info['n_select_bands']}, scale={args.scale_ratio}x, params={params:,}"
    )
    print(
        f"degradation={args.degradation_mode}; train misalignment={args.train_misalignment_mode} "
        f"translation<=±{args.translation_max_px:g}px rotation<=±{args.rotation_max_deg:g}deg "
        f"local<={args.local_max_displacement_px:g}px; mask={not args.no_mask}; amp={args.amp}"
    )

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        shift_sum = 0.0

        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            warped_msi, _, mis = warp_msi_for_protocol(
                hr_msi,
                mode=args.train_misalignment_mode,
                translation_max_px=args.translation_max_px,
                rotation_max_deg=args.rotation_max_deg,
                local_max_displacement_px=args.local_max_displacement_px,
                control_grid_size=args.control_grid_size,
                generator=train_warp_gen,
            )

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=args.amp and device.type == "cuda",
            ):
                pred, _ = model(lr_hsi, warped_msi)
                # Official HSIFN training uses SmoothL1 reconstruction loss only.
                loss = F.smooth_l1_loss(pred, gt)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            bs = int(gt.shape[0])
            loss_sum += float(loss.item()) * bs
            count += bs
            magnitude = torch.sqrt(mis.dx_px.square() + mis.dy_px.square()).mean().item()
            shift_sum += float(magnitude) * bs

        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={loss_sum / max(count, 1):.6f} "
            f"mean_train_shift={shift_sum / max(count, 1):.3f}px"
        )

        do_eval = (epoch + 1) % validation_interval == 0 or epoch + 1 == args.epochs
        if do_eval:
            val = validate(model, val_loader, device, args)
            print(
                f"  valid: PSNR={val['PSNR_valid']:.4f} SAM={val['SAM_valid']:.4f} "
                f"full_PSNR={val['PSNR']:.4f} valid_fraction={val['valid_fraction']:.4f} "
                f"flow={val['flow_mean_px']:.3f}px mask={val['mask_mean']:.3f}"
            )
            improved = val["PSNR_valid"] > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr = float(val["PSNR_valid"])
                bad_validations = 0
                torch.save(
                    checkpoint_payload(model, optimizer, epoch, best_psnr, args),
                    os.path.join(args.checkpoint_dir, "best.pth.tar"),
                )
                print(f"  saved best checkpoint: PSNR_valid={best_psnr:.4f}")
            else:
                bad_validations += 1
                print(f"  no significant improvement: {bad_validations}/{args.early_stop_patience}")
                if bad_validations >= args.early_stop_patience:
                    print("Early stopping triggered by validation PSNR.")
                    torch.save(
                        checkpoint_payload(model, optimizer, epoch, best_psnr, args),
                        os.path.join(args.checkpoint_dir, "last.pth.tar"),
                    )
                    break

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            torch.save(
                checkpoint_payload(model, optimizer, epoch, best_psnr, args),
                os.path.join(args.checkpoint_dir, "last.pth.tar"),
            )


if __name__ == "__main__":
    main()
