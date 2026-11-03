"""Train SA-RFM under comparison_experiments' registered / mixed x4 protocol.

Released SA-RFM network and loss implementations: upstream_sb.py.
The effective upstream objective is RFM + lambda_otd * OTD; the gradient
and spectral-angle terms are computed for logging upstream but omitted
from its actual total_loss, so they are not added here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from common import (  # noqa: E402
    add_data_args, add_geometry_args, build_shared_cfg, deformed_observation,
    geometry_kwargs, make_deformed_lr_hsi, make_generator,
    require_srf_weights, resolve_device, sample_synthetic_geometry, set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from experiment_protocol import DATASET_VALIDATION_INTERVAL  # noqa: E402
from model import ARCHITECTURE_ID, UPSTREAM_BLOB_SHA, build_model  # noqa: E402


def parser(stage="registered"):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--train_misalignment_mode", choices=["registered"], default="registered")
    p.add_argument("--epochs", type=int, default=800 if stage == "registered" else 600)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=1001 if stage == "registered" else 10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--validation_interval", type=int, default=None)
    p.add_argument("--validation_cases", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=999999)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--init_checkpoint", default="")
    p.add_argument("--dim", type=int, default=220)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=10.0)
    p.add_argument("--sigma", type=float, default=1.0)
    p.add_argument("--lambda_otd", type=float, default=1.0)
    p.add_argument("--ode_steps", type=int, default=1)
    p.add_argument("--grad_clip", type=float, default=5.0)
    p.add_argument("--max_train_batches", type=int, default=0,
                   help="Smoke-test only; 0 means the full training split")
    return p.parse_args()


def sensor_fingerprint(info):
    srf = require_srf_weights(info)
    return hashlib.sha256(srf.detach().cpu().numpy().tobytes()).hexdigest()


def snapshot_rng(geometry_gen):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "geometry": geometry_gen.get_state(),
    }


def restore_rng(rng, geometry_gen):
    random.setstate(rng["python"])
    np.random.set_state(rng["numpy"])
    torch.set_rng_state(rng["torch"].cpu())
    if rng.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in rng["cuda"]])
    if rng.get("geometry") is not None:
        geometry_gen.set_state(rng["geometry"].cpu())


def checkpoint_payload(model, optimizer, args, epoch, best_psnr, bad_validations,
                       fingerprint, geometry_gen):
    return {
        "method": "SA-RFM",
        "architecture_id": ARCHITECTURE_ID,
        "upstream_blob_sha": UPSTREAM_BLOB_SHA,
        "epoch": int(epoch),
        "training_stage": args.stage,
        "hsi_channels": model.hsi_channels,
        "msi_channels": model.msi_channels,
        "model_args": {
            "dim": model.dim, "alpha": model.alpha, "beta": model.beta,
            "sigma": model.sigma, "ode_steps": model.ode_steps,
        },
        "registered_probability": 1.0 if args.stage == "registered" else args.registered_probability,
        "sensor_fingerprint": fingerprint,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args).copy(),
        "best_psnr": float(best_psnr),
        "bad_validations": int(bad_validations),
        "rng": snapshot_rng(geometry_gen),
    }


def load_checkpoint(path, model, args, *, expected_stage, fingerprint, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("method") != "SA-RFM" or state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Checkpoint is not compatible with this SA-RFM architecture")
    if state.get("training_stage") != expected_stage:
        raise ValueError(f"Expected {expected_stage} checkpoint, got {state.get('training_stage')}")
    if state.get("sensor_fingerprint") != fingerprint:
        raise ValueError("Checkpoint uses a different sensor SRF")
    if state.get("hsi_channels") != model.hsi_channels or state.get("msi_channels") != model.msi_channels:
        raise ValueError("Checkpoint HSI/MSI channel counts do not match current sensor")
    expected_config = {
        "dim": model.dim, "alpha": model.alpha, "beta": model.beta,
        "sigma": model.sigma, "ode_steps": model.ode_steps,
    }
    if state.get("model_args") != expected_config:
        raise ValueError(f"SA-RFM architecture/inference mismatch: {state.get('model_args')} != {expected_config}")
    recorded = state.get("args", {})
    for name in ("dataset", "image_size", "patch_size", "stride", "scale_ratio",
                 "degradation_mode", "mtf_nyquist", "psf_truncate", "lambda_otd"):
        if name == "lambda_otd" and optimizer is None:
            continue  # Inference does not use the training OTD weight.
        if name in recorded and getattr(args, name, None) != recorded[name]:
            raise ValueError(f"SA-RFM checkpoint protocol mismatch for {name}")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        for name in ("seed", "lr", "weight_decay", "batch_size",
                     "validation_interval", "validation_cases", "max_train_batches"):
            if name in recorded and getattr(args, name, None) != recorded[name]:
                raise ValueError(f"Exact resume mismatch: {name}")
        if "optimizer" not in state or "rng" not in state:
            raise ValueError("Exact resume requires optimizer and RNG state")
        optimizer.load_state_dict(state["optimizer"])
    return state


@torch.no_grad()
def evaluate(model, loader, device, p0, args):
    model.eval()
    reg = MetricAverager()
    warp = MetricAverager()
    geom_gen = make_generator(device, args.seed + 60000)
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        msi = batch["hr_msi"].to(device, non_blocking=True)
        registered_lr = p0.degrade(gt)
        reg.update(calc_metrics(model(registered_lr, msi), gt, args.scale_ratio))
        if args.stage == "mixed":
            for _ in range(args.validation_cases):
                geometry = sample_synthetic_geometry(
                    *gt.shape[-2:],
                    device=device, dtype=gt.dtype, generator=geom_gen,
                    local_strength_min_fraction=0.0,
                    local_strength_max_fraction=1.0,
                    **geometry_kwargs(args),
                )
                lr_deformed = make_deformed_lr_hsi(gt, geometry, p0)
                warp.update(calc_metrics(model(lr_deformed, msi), gt, args.scale_ratio))
    return reg.average(), warp.average()


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main(stage="registered"):
    if stage not in ("registered", "mixed"):
        raise ValueError(f"Unsupported SA-RFM stage: {stage}")
    args = parser(stage)
    args.stage = stage
    args.validation_interval = args.validation_interval or DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs, args.batch_size, args.validation_interval, args.validation_cases,
           args.ode_steps, args.dim) < 1:
        raise ValueError("Epochs, batch_size, validation interval/cases, ode_steps and dim must be positive")
    if args.lambda_otd < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid optimizer/loss settings")
    if args.max_train_batches < 0:
        raise ValueError("--max_train_batches must be >= 0")
    if args.degradation_mode != "physical":
        raise ValueError("SA-RFM unified stages require the shared physical degradation")
    if stage == "registered" and args.init_checkpoint:
        raise ValueError("Registered stage starts from scratch; use --resume to continue")

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_set, val_set, _, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(train_set) or not len(val_set):
        raise ValueError("SA-RFM benchmark splits cannot be empty")
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    p0 = train_set.degradation_operator.to(device)
    fingerprint = sensor_fingerprint(info)
    model = build_model(info, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    family = "physical" if stage == "registered" else "hsi_warp_final"
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else THIS_DIR / "checkpoints" / family / args.dataset
    log_dir = Path(args.log_dir) if args.log_dir else THIS_DIR / "logs" / family / args.dataset
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    geometry_gen = make_generator(device, args.seed + 404)
    start_epoch, best_psnr, bad_validations = 0, -float("inf"), 0
    if args.resume:
        state = load_checkpoint(args.resume, model, args, expected_stage=stage,
                                fingerprint=fingerprint, optimizer=optimizer)
        start_epoch = state["epoch"] + 1
        best_psnr = state["best_psnr"]
        bad_validations = state.get("bad_validations", 0)
        restore_rng(state["rng"], geometry_gen)
    elif stage == "mixed":
        init_path = args.init_checkpoint or str(
            THIS_DIR / "checkpoints" / "physical" / args.dataset / "best.pth.tar"
        )
        state = load_checkpoint(init_path, model, args, expected_stage="registered",
                                fingerprint=fingerprint)
        print(f"Initialized mixed SA-RFM from registered checkpoint: {init_path}", flush=True)

    protocol = {
        "method": "SA-RFM", "architecture_id": ARCHITECTURE_ID,
        "upstream_blob_sha": UPSTREAM_BLOB_SHA,
        "dataset": args.dataset, "stage": stage,
        "hsi_channels": model.hsi_channels, "msi_channels": model.msi_channels,
        "sensor_fingerprint": fingerprint,
        "training_objective": "RFM + lambda_otd * OTD (official effective loss)",
        "optimizer": "AdamW", "args": vars(args),
    }
    (ckpt_dir / "run_protocol.txt").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sum_total, sum_rfm, sum_otd, count, identities = 0.0, 0.0, 0.0, 0, 0
        progress = tqdm(loader, desc=f"SA-RFM {stage} {epoch + 1}/{args.epochs}",
                        dynamic_ncols=True, mininterval=1.0)
        for batch_idx, batch in enumerate(progress):
            if args.max_train_batches and batch_idx >= args.max_train_batches:
                break
            gt = batch["gt"].to(device, non_blocking=True)
            msi = batch["hr_msi"].to(device, non_blocking=True)
            with torch.no_grad():
                lr_hsi, identity_count = deformed_observation(
                    gt, p0, args, geometry_gen, mixed=(stage == "mixed")
                )
            optimizer.zero_grad(set_to_none=True)
            loss, details = model.flow_loss(lr_hsi, gt, msi, lambda_otd=args.lambda_otd)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"SA-RFM nonfinite loss at epoch={epoch+1} batch={batch_idx}")
            loss.backward()
            if args.grad_clip > 0:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError(f"SA-RFM nonfinite gradient at epoch={epoch+1} batch={batch_idx}")
            optimizer.step()
            b = int(gt.shape[0])
            count += b
            identities += identity_count
            sum_total += float(loss.detach()) * b
            sum_rfm += float(details["rfm"]) * b
            sum_otd += float(details["otd"]) * b
            progress.set_postfix(loss=f"{sum_total/max(count,1):.5f}",
                                 identity=f"{identities/max(count,1):.3f}", refresh=False)

        mean = {
            "train_total": sum_total / max(count, 1),
            "train_rfm": sum_rfm / max(count, 1),
            "train_otd": sum_otd / max(count, 1),
            "identity_fraction": identities / max(count, 1),
        }
        print(f"SA-RFM {stage} epoch={epoch+1} {mean}", flush=True)
        evaluate_now = (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.epochs
        if evaluate_now:
            reg, warp = evaluate(model, val_loader, device, p0, args)
            monitor = warp if stage == "mixed" else reg
            score = float(monitor["PSNR"])
            improved = score > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr, bad_validations = score, 0
            else:
                bad_validations += 1
            record = {
                "epoch": epoch + 1, **mean,
                **{f"reg_{k}": v for k, v in reg.items()},
                **{f"warp_{k}": v for k, v in warp.items()},
                "monitored_PSNR": score, "best_PSNR": best_psnr,
            }
            append_csv(log_dir / "validation_history.csv", record)
            print(f"VALIDATION {record}", flush=True)
            if improved:
                torch.save(checkpoint_payload(
                    model, optimizer, args, epoch, best_psnr, bad_validations,
                    fingerprint, geometry_gen
                ), ckpt_dir / "best.pth.tar")
            if args.early_stop_patience > 0 and bad_validations >= args.early_stop_patience:
                print("SA-RFM validation early stop", flush=True)
                torch.save(checkpoint_payload(
                    model, optimizer, args, epoch, best_psnr, bad_validations,
                    fingerprint, geometry_gen
                ), ckpt_dir / "last.pth.tar")
                break

        torch.save(checkpoint_payload(
            model, optimizer, args, epoch, best_psnr, bad_validations,
            fingerprint, geometry_gen
        ), ckpt_dir / "last.pth.tar")


if __name__ == "__main__":
    main()
