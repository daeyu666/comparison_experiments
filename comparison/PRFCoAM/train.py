"""Train PRFCoAM on the comparison_experiments shared registered protocol.

Phase A intentionally keeps HR-MSI, LR-HSI and GT registered. The author's
`base/` implementation is preserved; dataset/channel/device adaptation lives in
`model_adapter.py`. Once this registered sanity check is healthy, the HR-MSI
misregistration protocol can be added without conflating reproduction bugs with
registration-direction changes.
"""

from __future__ import annotations

import argparse
import csv
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
    build_shared_cfg,
    checkpoint_payload,
    require_srf_weights,
    resolve_device,
    set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from model_adapter import (  # noqa: E402
    build_prfcoam,
    displacement_smoothness,
    unpack_outputs,
)


def parse_args():
    p = argparse.ArgumentParser(description="PRFCoAM registered reproduction")
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

    # Released PRFCoAM uses Adam lr=5e-4, batch=8, StepLR(100, 0.8), 3001 epochs.
    # We keep optimizer/scheduler values but use the shared benchmark validation
    # and early-stopping protocol, so a 300-epoch cap is sufficient for sanity runs.
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--scheduler_step", type=int, default=100)
    p.add_argument("--scheduler_gamma", type=float, default=0.8)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--device", default="cuda:0")

    p.add_argument("--validation_interval", type=int, default=0,
                   help="0 uses PaviaU=20, Houston13=10, Chikusei=5")
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--resume", default="")
    return p.parse_args()


def make_loader(dataset, batch_size: int, shuffle: bool, workers: int) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=True,
        drop_last=False,
    )


def load_resume(model, optimizer, scheduler, path: str, device: torch.device):
    if not path:
        return 0, float("-inf")
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"] if "model" in ckpt else ckpt)
    if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    if isinstance(ckpt, dict) and ckpt.get("scheduler") is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0
    best = float(ckpt.get("best_psnr", float("-inf"))) if isinstance(ckpt, dict) else float("-inf")
    print(f"Resumed {path}: start_epoch={start}, best_valid_PSNR={best:.4f}")
    return start, best


def official_loss(outputs, gt: torch.Tensor, hr_msi: torch.Tensor):
    """Released PRFCoAM objective, written without redundant recomputation."""
    pred, pred_msi, _pred_lrms, _reg_lrhs, rg1, rg2 = unpack_outputs(outputs)
    loss_recon = F.l1_loss(pred, gt)
    loss_sensor = F.l1_loss(pred_msi, hr_msi)
    loss_smooth = 0.5 * (
        displacement_smoothness(rg1) + displacement_smoothness(rg2)
    )
    # Released code: loss1 + 0.1*(loss_sensor + loss1) + 0.01*loss_smooth.
    loss = 1.1 * loss_recon + 0.1 * loss_sensor + 0.01 * loss_smooth
    return loss, loss_recon, loss_sensor, loss_smooth


@torch.no_grad()
def validate(model, loader, device, scale_ratio: int):
    model.eval()
    metric_rows = []
    recon_sum = 0.0
    flow1_sum = 0.0
    flow2_sum = 0.0
    count = 0
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        outputs = model(lr_hsi, hr_msi)
        pred, _pred_msi, _pred_lrms, _reg_lrhs, rg1, rg2 = unpack_outputs(outputs)
        pred_metric = pred.clamp(0.0, 1.0)
        metric_rows.append(calc_metrics(pred_metric, gt, scale_ratio))
        bs = int(gt.shape[0])
        recon_sum += float(F.l1_loss(pred, gt).item()) * bs
        flow1_sum += float(torch.linalg.vector_norm(rg1, dim=1).mean().item()) * bs
        flow2_sum += float(torch.linalg.vector_norm(rg2, dim=1).mean().item()) * bs
        count += bs

    mean_metrics = {
        key: sum(row[key] for row in metric_rows) / len(metric_rows)
        for key in metric_rows[0]
    }
    mean_metrics.update(
        L1=recon_sum / max(count, 1),
        RG1_mean_px=flow1_sum / max(count, 1),
        RG2_mean_px=flow2_sum / max(count, 1),
    )
    return mean_metrics


def append_history(path: Path, row: dict) -> None:
    write_header = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def main():
    args = parse_args()
    if args.scale_ratio != 4:
        raise ValueError("Released PRFCoAM topology has two x2 fusion stages and requires scale_ratio=4")
    if args.patch_size % 4 != 0 or args.image_size % 4 != 0:
        raise ValueError("PRFCoAM HR sizes must be divisible by 4")
    if (args.patch_size // 4) % 2 != 0 or (args.image_size // 4) % 2 != 0:
        raise ValueError("PRFCoAM LR spatial sizes must be even for the 2x2 spectral Mamba scan")

    set_seed(args.seed)
    cfg = build_shared_cfg(args)
    train_set, val_set, _, info = build_datasets(cfg, include_validation=True)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, 1, False, 0)

    device = resolve_device(args.device)
    if device.type != "cuda":
        raise RuntimeError(
            "The released PRFCoAM Mamba path imports selective_scan_cuda and "
            "causal_conv1d_cuda; this reproduction currently requires CUDA."
        )
    # Ensure the shared MSI really came from the fixed sensor SRF protocol.
    require_srf_weights(info)

    model = build_prfcoam(
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        device=device,
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=args.scheduler_step, gamma=args.scheduler_gamma
    )

    if not args.checkpoint_dir:
        args.checkpoint_dir = str(THIS_DIR / "checkpoints" / args.degradation_mode / args.dataset)
    if not args.log_dir:
        args.log_dir = str(THIS_DIR / "logs" / args.degradation_mode / args.dataset)
    checkpoint_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    validation_interval = args.validation_interval or {
        "PaviaU": 20,
        "Houston13": 10,
        "Chikusei": 5,
    }[args.dataset]

    protocol_path = checkpoint_dir / "run_protocol.txt"
    with protocol_path.open("w", encoding="utf-8") as f:
        f.write("method: PRFCoAM registered reproduction\n")
        f.write("author_base_preserved: comparison/PRFCoAM/base\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"degradation_mode: {args.degradation_mode}\n")
        f.write(f"srf_profile: {info.get('srf_profile')}\n")
        f.write(f"hsi_bands: {info['n_bands']}\n")
        f.write(f"msi_bands: {info['n_select_bands']}\n")
        f.write(f"scale_ratio: {args.scale_ratio}\n")
        f.write(f"train_patch_hr: {args.patch_size}\n")
        f.write(f"train_patch_lr: {args.patch_size // args.scale_ratio}\n")
        f.write(f"validation_hr: {args.image_size}\n")
        f.write("misalignment: registered (Phase A sanity check)\n")
        f.write("loss: 1.1*L1_HSI + 0.1*L1_MSI + 0.01*flow_smoothness\n")
        f.write(f"lr: {args.lr}\n")
        f.write(f"scheduler: StepLR({args.scheduler_step}, gamma={args.scheduler_gamma})\n")
        f.write(f"validation_interval: {validation_interval}\n")
        f.write(f"early_stop_min_delta: {args.early_stop_min_delta}\n")
        f.write(f"early_stop_patience: {args.early_stop_patience}\n")

    start_epoch, best_psnr = load_resume(model, optimizer, scheduler, args.resume, device)
    bad_validations = 0
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"PRFCoAM: dataset={args.dataset}, HSI={info['n_bands']}, "
        f"MSI={info['n_select_bands']}, HRpatch={args.patch_size}, "
        f"LRpatch={args.patch_size // args.scale_ratio}, params={params:,}"
    )
    print(
        f"registered sanity check; degradation={args.degradation_mode}; "
        f"lr={args.lr:g}; batch={args.batch_size}; validation_every={validation_interval}"
    )

    history_path = log_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_sum = recon_sum = sensor_sum = smooth_sum = 0.0
        count = 0
        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            outputs = model(lr_hsi, hr_msi)
            loss, loss_recon, loss_sensor, loss_smooth = official_loss(outputs, gt, hr_msi)
            loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            bs = int(gt.shape[0])
            total_sum += float(loss.item()) * bs
            recon_sum += float(loss_recon.item()) * bs
            sensor_sum += float(loss_sensor.item()) * bs
            smooth_sum += float(loss_smooth.item()) * bs
            count += bs

        scheduler.step()
        train_row = {
            "epoch": epoch + 1,
            "train_loss": total_sum / max(count, 1),
            "train_l1": recon_sum / max(count, 1),
            "train_sensor_l1": sensor_sum / max(count, 1),
            "train_smooth": smooth_sum / max(count, 1),
            "lr": optimizer.param_groups[0]["lr"],
        }
        print(
            f"Epoch {epoch + 1:03d}/{args.epochs} "
            f"loss={train_row['train_loss']:.6f} "
            f"L1={train_row['train_l1']:.6f} "
            f"MSI={train_row['train_sensor_l1']:.6f} "
            f"smooth={train_row['train_smooth']:.6f} "
            f"lr={train_row['lr']:.3e}"
        )

        do_eval = (epoch + 1) % validation_interval == 0 or epoch + 1 == args.epochs
        if do_eval:
            val = validate(model, val_loader, device, args.scale_ratio)
            print(
                f"  valid: PSNR={val['PSNR']:.4f} SAM={val['SAM']:.4f} "
                f"RMSE={val['RMSE']:.6f} SSIM={val['SSIM']:.6f} "
                f"RG1={val['RG1_mean_px']:.3f}px RG2={val['RG2_mean_px']:.3f}px"
            )
            history = {**train_row, **{f"val_{k}": v for k, v in val.items()}}
            append_history(history_path, history)

            improved = val["PSNR"] > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr = float(val["PSNR"])
                bad_validations = 0
                torch.save(
                    checkpoint_payload(model, optimizer, scheduler, epoch, best_psnr, args),
                    checkpoint_dir / "best.pth.tar",
                )
                print(f"  saved best checkpoint: PSNR={best_psnr:.4f}")
            else:
                bad_validations += 1
                print(f"  no significant improvement: {bad_validations}/{args.early_stop_patience}")
                if bad_validations >= args.early_stop_patience:
                    torch.save(
                        checkpoint_payload(model, optimizer, scheduler, epoch, best_psnr, args),
                        checkpoint_dir / "last.pth.tar",
                    )
                    print("Early stopping triggered by validation PSNR.")
                    break
        else:
            # Keep train-only epochs in the CSV as well, using blank validation fields.
            append_history(history_path, train_row)

        if (epoch + 1) % args.save_interval == 0 or epoch + 1 == args.epochs:
            torch.save(
                checkpoint_payload(model, optimizer, scheduler, epoch, best_psnr, args),
                checkpoint_dir / "last.pth.tar",
            )


if __name__ == "__main__":
    main()
