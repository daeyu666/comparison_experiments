"""Registered and mixed MoE-PNP training under the shared comparison protocol."""
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
HERE = THIS_DIR
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
_COMMON_SPEC = importlib.util.spec_from_file_location("moe_pnp_common", THIS_DIR / "common.py")
if _COMMON_SPEC is None or _COMMON_SPEC.loader is None:
    raise ImportError("Cannot load MoE-PNP common.py")
common = importlib.util.module_from_spec(_COMMON_SPEC)
sys.modules[_COMMON_SPEC.name] = common
_COMMON_SPEC.loader.exec_module(common)

from data_loader import build_datasets
from metrics import MetricAverager, calc_metrics


def parser(stage="registered"):
    p = argparse.ArgumentParser(description=__doc__)
    common.add_data_args(p)
    common.add_geometry_args(p)
    p.set_defaults(stage=stage)
    p.add_argument("--train_misalignment_mode", choices=["registered"], default="registered")
    p.add_argument("--epochs", type=int, default=800 if stage == "registered" else 600)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--seed", type=int, default=1001 if stage == "registered" else 10)
    p.add_argument("--validation_interval", type=int, default=None)
    p.add_argument("--validation_cases", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=999999)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--init_checkpoint", default="")
    return p


def _use_dsp(image, flow, spatial_transform):
    flow = flow.permute(0, 2, 3, 1)
    image = image.permute(0, 2, 3, 1)
    return spatial_transform(image, flow).permute(0, 3, 1, 2)


def _smoothing_loss(flow):
    dx = torch.abs(flow[:, :, :, 1:] - flow[:, :, :, :-1])
    dy = torch.abs(flow[:, :, 1:, :] - flow[:, :, :-1, :])
    return (torch.mean(dx * dx) + torch.mean(dy * dy)) / 2.0


class _LocalNCC:
    """Official MoE-PNP squared local cross-correlation objective (negative for minimization)."""
    def __init__(self, window=9):
        self.window = int(window)

    def loss(self, first, second):
        if first.shape != second.shape or first.ndim != 4 or first.shape[1] != 1:
            raise ValueError("MoE-PNP registration consistency expects equal N x 1 x H x W inputs")
        window = self.window
        filt = torch.ones((1, 1, window, window), device=first.device, dtype=first.dtype)
        padding = window // 2
        first2, second2, cross_product = first * first, second * second, first * second
        first_sum = F.conv2d(first, filt, padding=padding)
        second_sum = F.conv2d(second, filt, padding=padding)
        first2_sum = F.conv2d(first2, filt, padding=padding)
        second2_sum = F.conv2d(second2, filt, padding=padding)
        cross_sum = F.conv2d(cross_product, filt, padding=padding)
        window_size = float(window * window)
        first_mean = first_sum / window_size
        second_mean = second_sum / window_size
        cross = cross_sum - second_mean * first_sum - first_mean * second_sum + first_mean * second_mean * window_size
        first_var = first2_sum - 2 * first_mean * first_sum + first_mean * first_mean * window_size
        second_var = second2_sum - 2 * second_mean * second_sum + second_mean * second_mean * window_size
        corr_sq = cross * cross / (first_var * second_var + 1e-5)
        return -torch.mean(corr_sq)


def compute_moepnp_loss(model, lr_hsi, hr_msi, gt, blur, ncc):
    (
        pred_hr, pred_mid, intensity_mid, intensity_hr,
        msi_mid_gray, msi_hr_gray, flow_mid, flow_hr,
        flow_mid_init, flow_hr_init, spatial_mid, spatial_hr,
    ) = model.forward_full(lr_hsi, hr_msi)

    gt_mid = F.interpolate(blur(gt), scale_factor=0.5, mode="bilinear", align_corners=False)
    loss_reconstruction = 0.5 * F.l1_loss(pred_hr, gt) + 0.5 * F.l1_loss(pred_mid, gt_mid)

    msi_mid_registered = _use_dsp(msi_mid_gray, flow_mid, spatial_mid)
    msi_hr_registered = _use_dsp(msi_hr_gray.detach(), flow_hr, spatial_hr)
    intensity_mid_registered = _use_dsp(intensity_mid, flow_mid_init, spatial_mid)
    intensity_hr_registered = _use_dsp(intensity_hr.detach(), flow_hr_init, spatial_hr)

    loss_registration = (
        ncc.loss(intensity_mid, msi_mid_registered)
        + ncc.loss(intensity_hr.detach(), msi_hr_registered)
        + ncc.loss(msi_mid_gray, intensity_mid_registered)
        + ncc.loss(msi_hr_gray.detach(), intensity_hr_registered)
    ) / 4.0
    loss_smoothness = (
        _smoothing_loss(flow_mid) + _smoothing_loss(flow_hr)
        + _smoothing_loss(flow_mid_init) + _smoothing_loss(flow_hr_init)
    ) / 4.0
    total = loss_reconstruction + 0.001 * loss_registration + 0.01 * loss_smoothness
    return total, {
        "reconstruction": loss_reconstruction,
        "registration": loss_registration,
        "smoothness": loss_smoothness,
    }


@torch.no_grad()
def evaluate(model, loader, device, degradation, args):
    model.eval()
    registered_meter, warp_meter = MetricAverager(), MetricAverager()
    for batch in loader:
        gt = batch["gt"].to(device, dtype=torch.float32, non_blocking=True)
        hr_msi = batch["hr_msi"].to(device, dtype=torch.float32, non_blocking=True)
        registered_lr = degradation.degrade(gt)
        registered_pred = model(registered_lr, hr_msi)
        registered_meter.update(calc_metrics(registered_pred, gt, args.scale_ratio))

        if args.stage == "mixed":
            generator = common.make_generator(device, args.seed + 60000)
            for _ in range(args.validation_cases):
                geometry = common.sample_synthetic_geometry(
                    *gt.shape[-2:], device=device, dtype=gt.dtype, generator=generator,
                    local_strength_min_fraction=0.0, local_strength_max_fraction=1.0,
                    **common.geometry_kwargs(args),
                )
                warped_lr = common.make_deformed_lr_hsi(gt, geometry, degradation)
                warped_pred = model(warped_lr, hr_msi)
                warp_meter.update(calc_metrics(warped_pred, gt, args.scale_ratio))
    return registered_meter.average(), warp_meter.average()


def main(stage="registered"):
    args = parser(stage).parse_args()
    args.validation_interval = args.validation_interval or common.DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs, args.batch_size, args.validation_interval, args.validation_cases) < 1:
        raise ValueError("Training schedule values must be positive")
    if args.scale_ratio != 4:
        raise ValueError("MoE-PNP supports the shared synthetic x4 protocol only")
    if (args.image_size, args.patch_size, args.stride) != (128, 64, 32):
        raise ValueError("Unified synthetic protocol is fixed at image_size=128, patch_size=64, stride=32")
    if not 0.0 <= args.registered_probability <= 1.0:
        raise ValueError("registered_probability must be in [0,1]")
    if stage == "mixed" and args.degradation_mode != "physical":
        raise ValueError("Formal mixed training requires physical degradation")
    if stage == "registered" and args.init_checkpoint:
        raise ValueError("Registered training starts from scratch; use --resume to continue")

    common.set_seed(args.seed)
    device = common.resolve_device(args.device)
    train_set, val_set, _unused_test, info = build_datasets(
        common.build_shared_cfg(args), include_validation=True
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    degradation = train_set.degradation_operator.to(device)
    model = common.model_for_info(info, args, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    generator = common.make_generator(device, args.seed + 404)
    start_epoch, best_psnr, bad_validations = 0, float("-inf"), 0

    if args.resume:
        state = common.load_checkpoint(args.resume, model, args, expected_stage=stage, optimizer=optimizer)
        start_epoch = int(state["epoch"]) + 1
        best_psnr = float(state["best_psnr"])
        bad_validations = int(state["bad_validations"])
        common.restore_rng(state["rng"], generator)
        print(f"Resumed MoE-PNP {stage} from epoch {start_epoch}: {args.resume}")
    elif stage == "mixed":
        init_checkpoint = args.init_checkpoint or str(
            HERE / "checkpoints" / "physical" / args.dataset / "best.pth.tar"
        )
        common.load_checkpoint(init_checkpoint, model, args, expected_stage="registered")
        args.init_checkpoint = init_checkpoint
        print(f"Loaded registered MoE-PNP weights only; initialized a fresh AdamW optimizer: {init_checkpoint}")

    family = args.degradation_mode if stage == "registered" else "hsi_warp_final"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else HERE / "checkpoints" / family / args.dataset
    log_dir = Path(args.log_dir) if args.log_dir else HERE / "logs" / family / args.dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    protocol = {
        "method": "MoE-PNP",
        **common.architecture_metadata(model),
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "loss": "0.5*L1(HRHS,GT)+0.5*L1(MRHS,blur_down2(GT))+0.001*NCC+0.01*flow_smoothness",
        "observation": "Y_H=P0(W_phi(X)); HR-MSI=R0(X) remains in the reference coordinates; 10% identity + 90% deformed HSI in mixed stage",
        "validation_rect": info.get("validation_rect"),
        "srf_profile": info.get("srf_profile"),
        "sensor_fingerprint": getattr(model, "sensor_fingerprint", None),
        "arguments": vars(args).copy(),
    }
    (checkpoint_dir / "run_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    history_path = log_dir / "validation_history.csv"
    print(json.dumps(protocol, indent=2), flush=True)

    blur = transforms.GaussianBlur(kernel_size=15, sigma=1.0)
    ncc = _LocalNCC(window=9)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum = recon_sum = ncc_sum = smooth_sum = 0.0
        sample_count = identity_count = 0
        progress = tqdm(
            train_loader, desc=f"MoE-PNP {stage} {epoch + 1}/{args.epochs}",
            dynamic_ncols=True, mininterval=1.0,
        )
        for batch in progress:
            gt = batch["gt"].to(device, dtype=torch.float32, non_blocking=True)
            hr_msi = batch["hr_msi"].to(device, dtype=torch.float32, non_blocking=True)
            with torch.no_grad():
                lr_hsi, identities = common.deformed_observation(
                    gt, degradation, args, generator, mixed=(stage == "mixed")
                )
            optimizer.zero_grad(set_to_none=True)
            loss, components = compute_moepnp_loss(model, lr_hsi, hr_msi, gt, blur, ncc)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"Non-finite MoE-PNP loss at epoch {epoch + 1}")
            loss.backward()
            optimizer.step()

            batch_n = int(gt.shape[0])
            sample_count += batch_n
            identity_count += identities
            loss_sum += float(loss.item()) * batch_n
            recon_sum += float(components["reconstruction"].detach().item()) * batch_n
            ncc_sum += float(components["registration"].detach().item()) * batch_n
            smooth_sum += float(components["smoothness"].detach().item()) * batch_n
            progress.set_postfix(
                loss=f"{loss_sum / max(sample_count, 1):.5f}",
                recon=f"{recon_sum / max(sample_count, 1):.5f}",
                ncc=f"{ncc_sum / max(sample_count, 1):.5f}",
                smooth=f"{smooth_sum / max(sample_count, 1):.5f}",
                identity=f"{identity_count / max(sample_count, 1):.3f}",
                refresh=False,
            )

        current_epoch = epoch + 1
        mean_loss = loss_sum / max(sample_count, 1)
        identity_fraction = identity_count / max(sample_count, 1)
        print(
            f"MoE-PNP {stage} epoch={current_epoch}/{args.epochs} "
            f"loss={mean_loss:.6f} identity_fraction={identity_fraction:.3f}",
            flush=True,
        )
        do_eval = current_epoch % args.validation_interval == 0 or current_epoch == args.epochs
        row = {
            "epoch": current_epoch,
            "train_loss": mean_loss,
            "train_reconstruction": recon_sum / max(sample_count, 1),
            "train_registration": ncc_sum / max(sample_count, 1),
            "train_smoothness": smooth_sum / max(sample_count, 1),
            "identity_fraction": identity_fraction,
        }

        if do_eval:
            registered, warped = evaluate(model, val_loader, device, degradation, args)
            row.update({"reg_" + key: value for key, value in registered.items()})
            row.update({"warp_" + key: value for key, value in warped.items()})
            score = warped["PSNR"] if stage == "mixed" else registered["PSNR"]
            improved = score > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr, bad_validations = float(score), 0
            else:
                bad_validations += 1

            write_header = not history_path.exists()
            with history_path.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
                if write_header:
                    writer.writeheader()
                writer.writerow(row)

            print(
                f"VALIDATION {row} best_PSNR={best_psnr:.5f} bad_validations={bad_validations}",
                flush=True,
            )
            if improved:
                torch.save(
                    common.checkpoint_payload(
                        model, optimizer, epoch, best_psnr, bad_validations, args, generator
                    ),
                    checkpoint_dir / "best.pth.tar",
                )

        torch.save(
            common.checkpoint_payload(
                model, optimizer, epoch, best_psnr, bad_validations, args, generator
            ),
            checkpoint_dir / "last.pth.tar",
        )
        if do_eval and args.early_stop_patience > 0 and bad_validations >= args.early_stop_patience:
            print(f"Validation early stopping at epoch {current_epoch}", flush=True)
            break


if __name__ == "__main__":
    main()
