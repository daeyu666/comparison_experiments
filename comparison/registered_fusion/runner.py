"""Reuse the public registered protocol for supervised LR-HSI/HR-MSI fusion."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import math
import random
import sys
from itertools import islice
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from comparison.UAFL.common import build_shared_cfg, require_srf_weights, set_seed
from data_loader import build_datasets
from experiment_protocol import (
    BATCH_SIZE, DATASETS, DATASET_VALIDATION_INTERVAL, DEGRADATION_MODE, EARLY_STOP_MIN_DELTA,
    EARLY_STOP_PATIENCE_FIXED_BUDGET, EVAL_SEED, IMAGE_SIZE, MTF_NYQUIST, PATCH_SIZE,
    PSF_TRUNCATE, REGISTERED_EPOCHS, REGISTERED_NEURAL_OPTIMIZERS, REGISTERED_SEED, ROOT,
    SCALE_RATIO, STRIDE, resolve_model,
)
from metrics import MetricAverager, calc_metrics


PROTOCOL = {
    "image_size": IMAGE_SIZE, "patch_size": PATCH_SIZE, "stride": STRIDE,
    "scale_ratio": SCALE_RATIO, "mtf_nyquist": MTF_NYQUIST, "psf_truncate": PSF_TRUNCATE,
    "degradation_sigma": 2.0, "degradation_kernel_size": 5,
}


def parser(method):
    p = argparse.ArgumentParser(description=f"{method} on the shared registered ×4 benchmark")
    p.add_argument("--dataset", choices=DATASETS, default="PaviaU")
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--degradation_mode", choices=("physical", "gaussian_bicubic"), default=DEGRADATION_MODE)
    for name, value in PROTOCOL.items():
        p.add_argument("--" + name, type=type(value), choices=(value,), default=value)
    return p


def run_dir(adapter, kind, args):
    return ROOT / "comparison" / adapter.folder / kind / args.degradation_mode / args.dataset


def make_model(method, args, info):
    module = importlib.import_module(f"comparison.{method}.model")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu for a smoke check")
    model = module.build_model(info).to(device)
    srf = require_srf_weights(info).contiguous()
    metadata = {
        "method": method, "architecture_id": module.ARCHITECTURE_ID,
        "upstream_commit": module.UPSTREAM_COMMIT, "stage": "registered",
        "dataset": args.dataset, "protocol": {**PROTOCOL, "degradation_mode": args.degradation_mode},
        "srf_sha256": hashlib.sha256(srf.numpy().tobytes()).hexdigest(),
        "hsi_channels": model.hsi_channels, "msi_channels": model.msi_channels,
    }
    return model, device, metadata


def load_state(path, model, metadata):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for key, value in metadata.items():
        if state.get("metadata", {}).get(key) != value:
            raise ValueError(f"Checkpoint {path} differs for {key}")
    model.load_state_dict(state["model"], strict=True)
    return state


def snapshot_rng(device):
    np_state = np.random.get_state()
    return {
        "torch": torch.get_rng_state(), "python": random.getstate(),
        "numpy": (np_state[0], np_state[1].tolist(), *np_state[2:]),
        "cuda": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
    }


def restore_rng(state, device):
    torch.set_rng_state(state["torch"])
    random.setstate(state["python"])
    algorithm, values, *rest = state["numpy"]
    np.random.set_state((algorithm, np.asarray(values, dtype=np.uint32), *rest))
    if device.type == "cuda" and state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"], device)


def save_state(state, path):
    pending = path.with_suffix(".tmp")
    torch.save(state, pending)
    pending.replace(path)


def append_metrics(path, row):
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def batch_to_device(batch, device):
    return tuple(batch[key].to(device, non_blocking=True) for key in ("lr_hsi", "hr_msi", "gt"))


@torch.inference_mode()
def score(model, loader, device, *, keep_rows=False):
    model.eval()
    average = MetricAverager()
    rows = []
    for idx, batch in enumerate(loader, 1):
        lr, msi, gt = batch_to_device(batch, device)
        values = calc_metrics(model(lr, msi), gt, SCALE_RATIO)
        if not all(math.isfinite(v) for v in values.values()):
            raise FloatingPointError(f"Nonfinite registered evaluation metric at sample={idx}: {values}")
        average.update(values)
        if keep_rows:
            rows.append({"sample": idx, **values})
    return average.average(), rows


def train_main(method, argv=None):
    lr, weight_decay, milestones = REGISTERED_NEURAL_OPTIMIZERS[method]
    adapter = resolve_model(method)
    p = parser(method)
    p.add_argument("--train_misalignment_mode", choices=["registered"], default="registered")
    p.add_argument("--epochs", type=int, default=REGISTERED_EPOCHS)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=lr)
    p.add_argument("--weight_decay", type=float, default=weight_decay)
    p.add_argument("--seed", type=int, default=REGISTERED_SEED)
    p.add_argument("--validation_interval", type=int)
    p.add_argument("--early_stop_patience", type=int, default=EARLY_STOP_PATIENCE_FIXED_BUDGET)
    p.add_argument("--early_stop_min_delta", type=float, default=EARLY_STOP_MIN_DELTA)
    p.add_argument("--checkpoint_dir", type=Path)
    p.add_argument("--log_dir", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--max_train_batches", type=int, default=0, help="For isolated smoke runs only")
    args = p.parse_args(argv)
    if args.validation_interval is None:
        args.validation_interval = DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs, args.batch_size, args.validation_interval) < 1:
        p.error("epochs, batch_size and validation_interval must be positive")
    if args.lr <= 0 or min(args.weight_decay, args.num_workers, args.early_stop_patience,
                           args.early_stop_min_delta, args.max_train_batches) < 0:
        p.error("Invalid optimizer, early-stop or loader settings")
    checkpoints = args.checkpoint_dir or run_dir(adapter, "checkpoints", args)
    logs = args.log_dir or run_dir(adapter, "logs", args)
    if args.max_train_batches and (not args.checkpoint_dir or not args.log_dir):
        p.error("max_train_batches requires separate checkpoint_dir and log_dir")
    if not args.resume and ((checkpoints / "last.pth.tar").exists() or (checkpoints / "best.pth.tar").exists()
                            or (logs / "validation_history.csv").exists()):
        p.error("Existing run: use --resume or choose new checkpoint_dir and log_dir")
    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)
    train, val, _, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(train) or not len(val):
        raise ValueError("Empty registered training or validation split")
    model, device, metadata = make_model(method, args, info)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val, batch_size=1)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = (torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones, gamma=0.5)
                 if milestones else None)
    settings = {k: getattr(args, k) for k in (
        "lr", "weight_decay", "seed", "batch_size", "num_workers", "validation_interval",
        "early_stop_patience", "early_stop_min_delta", "max_train_batches",
    )}
    start = bad = 0
    best = significant_best = -math.inf
    if args.resume:
        state = load_state(args.resume, model, metadata)
        if state["settings"] != settings:
            raise ValueError("Resume settings differ from checkpoint (epochs may be extended)")
        start, best, significant_best, bad = (state[k] for k in (
            "epoch", "best_psnr", "significant_best_psnr", "bad_validations",
        ))
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(state["scheduler"])
        restore_rng(state["rng"], device)
    print(f"RUN_PROTOCOL {json.dumps({**metadata, **settings, 'epochs': args.epochs, 'train_samples': len(train), 'validation_samples': len(val)})}",
          flush=True)
    steps = min(args.max_train_batches or len(train_loader), len(train_loader))
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        used_lr = optimizer.param_groups[0]["lr"]
        total, samples = 0.0, 0
        with tqdm(islice(train_loader, steps), total=steps, desc=f"{method} {epoch}/{args.epochs}",
                  disable=not sys.stderr.isatty(), mininterval=1) as batches:
            for idx, batch in enumerate(batches, 1):
                lr, msi, gt = batch_to_device(batch, device)
                optimizer.zero_grad(set_to_none=True)
                loss = F.l1_loss(model(lr, msi), gt)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"{method} nonfinite loss at epoch={epoch} batch={idx}")
                loss.backward()
                optimizer.step()
                total += float(loss.detach()) * gt.shape[0]
                samples += gt.shape[0]
        train_l1 = total / samples
        if scheduler is not None:
            scheduler.step()
        improved = stop = False
        if epoch % args.validation_interval == 0 or epoch == args.epochs:
            metrics, _ = score(model, val_loader, device)
            psnr = metrics["PSNR"]
            if psnr > best:
                best, improved = psnr, True
            if psnr > significant_best + args.early_stop_min_delta:
                significant_best, bad = psnr, 0
            else:
                bad += 1
            row = {"epoch": epoch, "train_L1": train_l1, "lr": used_lr,
                   **{f"reg_{key}": value for key, value in metrics.items()}, "best_PSNR": best}
            append_metrics(logs / "validation_history.csv", row)
            print(f"VALIDATION {json.dumps(row)}", flush=True)
            stop = bool(args.early_stop_patience and bad >= args.early_stop_patience)
        else:
            print(f"EPOCH {epoch}/{args.epochs} train_L1={train_l1:.7f} lr={used_lr:g}", flush=True)
        state = {
            "metadata": metadata, "settings": settings, "epoch": epoch, "best_psnr": best,
            "significant_best_psnr": significant_best, "bad_validations": bad,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "scheduler": None if scheduler is None else scheduler.state_dict(),
            "rng": snapshot_rng(device),
        }
        if improved:
            save_state(state, checkpoints / "best.pth.tar")
        save_state(state, checkpoints / "last.pth.tar")
        if stop:
            print(f"EARLY_STOP epoch={epoch} validations_without_improvement={bad}", flush=True)
            break


def test_main(method, argv=None):
    adapter = resolve_model(method)
    p = parser(method)
    p.add_argument("--test_mode", choices=["registered_only"], default="registered_only")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output_json", type=Path)
    p.add_argument("--eval_seed", type=int, default=EVAL_SEED)
    p.add_argument("--max_samples", type=int, default=0)
    args = p.parse_args(argv)
    if args.max_samples < 0:
        p.error("max_samples must be nonnegative")
    set_seed(args.eval_seed)
    checkpoint = args.checkpoint or run_dir(adapter, "checkpoints", args) / "best.pth.tar"
    _, _, test, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(test):
        raise ValueError("Empty registered test split")
    model, device, metadata = make_model(method, args, info)
    state = load_state(checkpoint, model, metadata)
    loader = DataLoader(test, batch_size=1)
    selected = islice(loader, args.max_samples) if args.max_samples else loader
    scores, rows = score(model, selected, device, keep_rows=True)
    result = {
        **metadata, "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": state["epoch"],
        "test_samples": len(rows), "smoke_test": bool(args.max_samples or state["settings"]["max_train_batches"]),
        "registered": scores, "per_sample": rows, "eval_seed": args.eval_seed,
    }
    default_name = "smoke_summary.json" if result["smoke_test"] else "summary.json"
    output = args.output_json or run_dir(adapter, "outputs", args) / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print("FINAL_REGISTERED " + " ".join(f"{k}={v:.6f}" for k, v in scores.items()), flush=True)
    print(f"FINAL_TEST_JSON={output}", flush=True)
    return result
