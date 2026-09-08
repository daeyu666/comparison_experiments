"""Train paper-faithful UAFL on the shared HSI-MSI comparison protocol.

UAFL consumes an upsampled LR-HSI and an HR reference on the same HR grid.
The benchmark therefore generates the fixed LR-HSI with the shared degradation,
bicubically upsamples it to the HR grid (as in the paper), and uses the SRF MSI
as the reference.  In non-registration modes ONLY the HR-MSI reference is warped;
GT-HSI and LR-HSI remain fixed.
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

from common import (  # noqa: E402
    build_shared_cfg,
    checkpoint_payload,
    require_srf_weights,
    resolve_device,
    set_seed,
)
from data_loader import build_datasets  # noqa: E402
from degradations.misalignment import make_misaligned_msi  # noqa: E402
from metrics import calc_metrics  # noqa: E402
from model import (  # noqa: E402
    PAPER_DIM,
    PAPER_NUM_BLOCKS,
    PAPER_NUM_ENDMEMBERS,
    PAPER_PARAMETER_M,
    PAPER_STAGE,
    build_uafl,
    parameter_count,
)


def parse_args():
    p = argparse.ArgumentParser(description="UAFL CVPR-2026 reproduction")
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

    # Paper: AdamW, lr 1e-5, wd 5e-5, batch 1, L1; 150 ICVL / 300 REAL.
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=1001)
    p.add_argument("--device", default="cuda:0")

    p.add_argument(
        "--train_misalignment_mode",
        default="registered",
        choices=["registered", "translation", "global", "local", "global_local"],
        help="Only HR-MSI is warped; registered is the Phase-A sanity run.",
    )
    p.add_argument("--translation_max_px", type=float, default=6.0)
    p.add_argument("--rotation_max_deg", type=float, default=3.0)
    p.add_argument("--local_max_displacement_px", type=float, default=3.0)
    p.add_argument("--control_grid_size", type=int, default=5)
    p.add_argument("--valid_threshold", type=float, default=0.999)

    p.add_argument("--validation_interval", type=int, default=0,
                   help="0 uses shared PaviaU=20, Houston13=10, Chikusei=5")
    p.add_argument("--early_stop_patience", type=int, default=2)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--resume", default="")
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


def upsample_lr_hsi(lr_hsi: torch.Tensor, hr_size) -> torch.Tensor:
    """Paper input X^up: bicubic LR-HSI interpolation to the target HR grid."""
    return F.interpolate(
        lr_hsi,
        size=tuple(hr_size),
        mode="bicubic",
        align_corners=False,
    )


def _mode_limits(args):
    mode = args.train_misalignment_mode
    if mode == "registered":
        return 0.0, 0.0, 0.0
    if mode == "translation":
        return args.translation_max_px, 0.0, 0.0
    if mode == "global":
        return args.translation_max_px, args.rotation_max_deg, 0.0
    if mode == "local":
        return 0.0, 0.0, args.local_max_displacement_px
    if mode == "global_local":
        return args.translation_max_px, args.rotation_max_deg, args.local_max_displacement_px
    raise ValueError(mode)


def prepare_reference(hr_msi, args, generator=None):
    tmax, rmax, lmax = _mode_limits(args)
    if tmax == 0 and rmax == 0 and lmax == 0:
        valid = torch.ones(
            hr_msi.shape[0], 1, hr_msi.shape[2], hr_msi.shape[3],
            device=hr_msi.device, dtype=hr_msi.dtype,
        )
        return hr_msi, valid
    warped, valid, _params = make_misaligned_msi(
        hr_msi,
        translation_max_px=tmax,
        rotation_max_deg=rmax,
        local_max_displacement_px=lmax,
        control_grid_size=args.control_grid_size,
        generator=generator,
    )
    return warped, valid


def masked_psnr_sam(pred, target, valid, threshold=0.999):
    pred = pred.detach().float().clamp(0.0, 1.0)
    target = target.detach().float().clamp(0.0, 1.0)
    mask = valid.detach().float()[:, 0] >= float(threshold)
    n_pix = int(mask.sum().item())
    if n_pix < 1:
        return float("nan"), float("nan"), 0

    mask_c = mask.unsqueeze(1).expand_as(pred)
    mse = ((pred - target).square()[mask_c]).mean().item()
    psnr = 100.0 if mse <= 1e-12 else 10.0 * math.log10(1.0 / mse)

    dot = (pred * target).sum(dim=1)
    pn = torch.linalg.vector_norm(pred, dim=1)
    tn = torch.linalg.vector_norm(target, dim=1)
    spectral_valid = mask & (pn > 1e-12) & (tn > 1e-12)
    if spectral_valid.any():
        cos = (
            dot[spectral_valid]
            / (pn[spectral_valid] * tn[spectral_valid]).clamp_min(1e-12)
        ).clamp(-1.0, 1.0)
        sam = (torch.acos(cos) * 180.0 / math.pi).mean().item()
    else:
        sam = 0.0
    return psnr, sam, n_pix


@torch.no_grad()
def validate(model, loader, device, args):
    model.eval()
    rows = []
    valid_psnr, valid_sam, valid_pixels = [], [], []
    # Recreate the same validation deformation every evaluation.
    val_gen = torch.Generator(device="cpu")
    val_gen.manual_seed(1234)
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, non_blocking=True)
        ref, valid = prepare_reference(hr_msi, args, val_gen)
        x_up = upsample_lr_hsi(lr_hsi, gt.shape[-2:])
        pred = model(x_up, ref)
        rows.append(calc_metrics(pred.clamp(0, 1), gt, args.scale_ratio))
        p, s, n = masked_psnr_sam(pred, gt, valid, args.valid_threshold)
        valid_psnr.append(p)
        valid_sam.append(s)
        valid_pixels.append(n)

    out = {k: sum(r[k] for r in rows) / len(rows) for k in rows[0]}
    out["PSNR_valid"] = sum(valid_psnr) / len(valid_psnr)
    out["SAM_valid"] = sum(valid_sam) / len(valid_sam)
    out["valid_pixels"] = sum(valid_pixels)
    return out


def load_resume(model, optimizer, path, device):
    if not path:
        return 0, float("-inf")
    ckpt = torch.load(path, map_location=device)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state)
    if isinstance(ckpt, dict) and ckpt.get("optimizer") is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
    start = int(ckpt.get("epoch", -1)) + 1 if isinstance(ckpt, dict) else 0
    best = float(ckpt.get("best_psnr", float("-inf"))) if isinstance(ckpt, dict) else float("-inf")
    print(f"Resumed {path}: start_epoch={start}, best_PSNR={best:.4f}")
    return start, best


HISTORY_FIELDS = [
    "epoch", "train_l1", "lr", "val_PSNR", "val_SAM", "val_RMSE",
    "val_ERGAS", "val_SSIM", "val_CC", "val_PSNR_valid", "val_SAM_valid",
    "val_valid_pixels",
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
    if args.scale_ratio != 4:
        print(
            f"Note: paper supports multiple scale factors; shared benchmark currently requested x{args.scale_ratio}."
        )
    if args.patch_size % 8 or args.image_size % 8:
        raise ValueError("UAFL SACA uses 8x8 windows; HR patch/validation sizes must be divisible by 8")

    set_seed(args.seed)
    cfg = build_shared_cfg(args)
    train_set, val_set, _test_set, info = build_datasets(cfg, include_validation=True)
    require_srf_weights(info)
    train_loader = make_loader(train_set, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_set, 1, False, 0)

    device = resolve_device(args.device)
    model = build_uafl(int(info["n_select_bands"])).to(device)
    params = parameter_count(model)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    if not args.checkpoint_dir:
        tag = args.degradation_mode
        if args.train_misalignment_mode != "registered":
            tag += f"_{args.train_misalignment_mode}"
        args.checkpoint_dir = str(THIS_DIR / "checkpoints" / tag / args.dataset)
    if not args.log_dir:
        tag = args.degradation_mode
        if args.train_misalignment_mode != "registered":
            tag += f"_{args.train_misalignment_mode}"
        args.log_dir = str(THIS_DIR / "logs" / tag / args.dataset)
    ckpt_dir = Path(args.checkpoint_dir)
    log_dir = Path(args.log_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    validation_interval = args.validation_interval or {
        "PaviaU": 20, "Houston13": 10, "Chikusei": 5
    }[args.dataset]

    with (ckpt_dir / "run_protocol.txt").open("w", encoding="utf-8") as f:
        f.write("method: UAFL (CVPR 2026)\n")
        f.write("architecture: paper-faithful; only reference channel count adapted\n")
        f.write(f"paper_dim: {PAPER_DIM}\n")
        f.write(f"paper_stage: {PAPER_STAGE}\n")
        f.write(f"paper_num_blocks: {list(PAPER_NUM_BLOCKS)}\n")
        f.write(f"paper_num_endmembers: {PAPER_NUM_ENDMEMBERS}\n")
        f.write(f"model_params: {params} ({params/1e6:.4f}M)\n")
        f.write(f"paper_reported_params: {PAPER_PARAMETER_M:.2f}M\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"hsi_bands: {info['n_bands']}\n")
        f.write(f"reference_msi_bands: {info['n_select_bands']}\n")
        f.write(f"degradation: {args.degradation_mode}\n")
        f.write(f"srf_profile: {info.get('srf_profile')}\n")
        f.write(f"HR_train_patch: {args.patch_size}\n")
        f.write(f"LR_train_patch: {args.patch_size // args.scale_ratio}\n")
        f.write("UAFL_input: bicubic-upsampled LR-HSI on HR grid\n")
        f.write(f"misalignment_mode: {args.train_misalignment_mode}\n")
        f.write("misalignment_reference: HR-MSI only; GT-HSI/LR-HSI fixed\n")
        f.write(f"translation_max_px: {args.translation_max_px}\n")
        f.write(f"rotation_max_deg: {args.rotation_max_deg}\n")
        f.write(f"local_max_displacement_px: {args.local_max_displacement_px}\n")
        f.write(f"control_grid_size: {args.control_grid_size}\n")
        f.write(f"optimizer: AdamW(lr={args.lr}, weight_decay={args.weight_decay})\n")
        f.write("loss: L1 (paper)\n")
        f.write(f"validation_interval: {validation_interval}\n")
        f.write(f"early_stop_min_delta: {args.early_stop_min_delta}\n")
        f.write(f"early_stop_patience: {args.early_stop_patience}\n")

    start_epoch, best_psnr = load_resume(model, optimizer, args.resume, device)
    bad_validations = 0
    train_gen = torch.Generator(device="cpu")
    train_gen.manual_seed(args.seed + 404)

    print(
        f"UAFL: dataset={args.dataset} HSI={info['n_bands']} MSI={info['n_select_bands']} "
        f"HRpatch={args.patch_size} LRpatch={args.patch_size//args.scale_ratio} "
        f"params={params:,} ({params/1e6:.3f}M; paper reports ~{PAPER_PARAMETER_M:.2f}M)"
    )
    print(
        f"mode={args.train_misalignment_mode} degradation={args.degradation_mode} "
        f"AdamW lr={args.lr:g} wd={args.weight_decay:g} batch={args.batch_size}"
    )

    history_path = log_dir / "history.csv"
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = 0.0
        count = 0
        for batch in train_loader:
            gt = batch["gt"].to(device, non_blocking=True)
            lr_hsi = batch["lr_hsi"].to(device, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, non_blocking=True)
            ref, _valid = prepare_reference(hr_msi, args, train_gen)
            x_up = upsample_lr_hsi(lr_hsi, gt.shape[-2:])

            optimizer.zero_grad(set_to_none=True)
            pred = model(x_up, ref)
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
        print(
            f"Epoch {epoch+1:03d}/{args.epochs} L1={row['train_l1']:.6f} "
            f"lr={row['lr']:.3e}"
        )

        do_eval = (epoch + 1) % validation_interval == 0 or epoch + 1 == args.epochs
        if do_eval:
            val = validate(model, val_loader, device, args)
            print(
                f"  valid: PSNR={val['PSNR']:.4f} SAM={val['SAM']:.4f} "
                f"RMSE={val['RMSE']:.6f} SSIM={val['SSIM']:.6f} | "
                f"valid-overlap PSNR={val['PSNR_valid']:.4f} "
                f"SAM={val['SAM_valid']:.4f}"
            )
            for k, v in val.items():
                row[f"val_{k}"] = v
            append_history(history_path, row)

            monitor = val["PSNR_valid"]
            improved = monitor > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr = float(monitor)
                bad_validations = 0
                torch.save(
                    checkpoint_payload(model, optimizer, epoch, best_psnr, args),
                    ckpt_dir / "best.pth.tar",
                )
                print(f"  best updated: PSNR_valid={best_psnr:.4f}")
            else:
                bad_validations += 1
                print(
                    f"  no improvement >= {args.early_stop_min_delta:.3f} dB "
                    f"({bad_validations}/{args.early_stop_patience})"
                )

            torch.save(
                checkpoint_payload(model, optimizer, epoch, best_psnr, args),
                ckpt_dir / "last.pth.tar",
            )
            if bad_validations >= args.early_stop_patience:
                print(f"Early stopping at epoch {epoch+1}; best PSNR_valid={best_psnr:.4f}")
                break
        elif (epoch + 1) % args.save_interval == 0:
            torch.save(
                checkpoint_payload(model, optimizer, epoch, best_psnr, args),
                ckpt_dir / "last.pth.tar",
            )


if __name__ == "__main__":
    main()
