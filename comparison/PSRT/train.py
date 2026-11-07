"""Train the 2023 PSRT under the common synthetic HSI/MSI comparison protocol.

Network: original 2x3 pyramid shuffle-and-reshuffle Transformer.
Upstream: https://github.com/shangqideng/PSRT (model/model_SR.py)
Optimizer/loss: AdamW 1e-4/1e-4, L1 reconstruction, clip_grad_norm_=10.
Data: identical common x4 registered/mixed split, SRF, PSF and geometry.
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for folder in (str(ROOT), str(HERE)):
    if folder not in sys.path:
        sys.path.insert(0, folder)

from common import (  # noqa: E402
    add_data_args, add_geometry_args, build_shared_cfg, deformed_observation,
    geometry_kwargs, make_deformed_lr_hsi, make_generator, require_srf_weights,
    resolve_device, sample_synthetic_geometry, set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from experiment_protocol import DATASET_VALIDATION_INTERVAL  # noqa: E402
from model import ARCHITECTURE_ID, UPSTREAM_SOURCE_SHA, build_model  # noqa: E402


def parse_args(stage="registered"):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--train_misalignment_mode", choices=["registered"], default="registered")
    p.add_argument("--epochs", type=int, default=800 if stage == "registered" else 600)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--grad_clip", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=1001 if stage == "registered" else 10)
    p.add_argument("--validation_interval", type=int, default=None)
    p.add_argument("--validation_cases", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=999999)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--init_checkpoint", default="")
    p.add_argument("--max_train_batches", type=int, default=0, help="Only for separate smoke runs")
    return p.parse_args()


def sensor_fingerprint(info):
    return hashlib.sha256(
        require_srf_weights(info).detach().cpu().numpy().tobytes()
    ).hexdigest()


def rng_snapshot(geometry_gen):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "geometry": geometry_gen.get_state(),
    }


def rng_restore(state, geometry_gen):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in state["cuda"]])
    geometry_gen.set_state(state["geometry"].cpu())


def checkpoint_payload(model, optimizer, args, epoch, best, bad, fingerprint, geometry_gen):
    return {
        "method": "PSRT",
        "architecture_id": ARCHITECTURE_ID,
        "upstream_source_sha": UPSTREAM_SOURCE_SHA,
        "epoch": int(epoch),
        "training_stage": args.stage,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "args": vars(args).copy(),
        "hsi_channels": model.hsi_channels,
        "msi_channels": model.msi_channels,
        "model_args": {"patch_size": model.patch_size, "embed_dim": 32,
                       "heads": 8, "outer_blocks": 2, "pyramid_blocks": 3,
                       "window_sizes": [8, 4, 2]},
        "sensor_fingerprint": fingerprint,
        "registered_probability": 1.0 if args.stage == "registered" else args.registered_probability,
        "best_psnr": float(best),
        "bad_validations": int(bad),
        "rng": rng_snapshot(geometry_gen),
    }


def load_checkpoint(path, model, args, *, expected_stage, fingerprint, optimizer=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("method") != "PSRT" or state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Incompatible PSRT checkpoint architecture")
    if state.get("training_stage") != expected_stage:
        raise ValueError(f"PSRT checkpoint stage is {state.get('training_stage')}, expected {expected_stage}")
    if state.get("sensor_fingerprint") != fingerprint:
        raise ValueError("PSRT checkpoint SRF differs from selected dataset")
    if (state.get("hsi_channels"), state.get("msi_channels")) != (
        model.hsi_channels, model.msi_channels
    ):
        raise ValueError("PSRT checkpoint HSI/MSI channel mismatch")
    if state.get("model_args", {}).get("patch_size") != model.patch_size:
        raise ValueError("PSRT checkpoint patch size mismatch")
    recorded = state.get("args", {})
    for field in ("dataset", "image_size", "patch_size", "stride", "scale_ratio",
                  "degradation_mode", "mtf_nyquist", "psf_truncate", "msi_mode",
                  "srf_interp", "srf_band_set"):
        if field in recorded and getattr(args, field, None) != recorded[field]:
            raise ValueError(f"Checkpoint/test protocol mismatch for {field}")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        for field in ("seed", "lr", "weight_decay", "batch_size", "grad_clip",
                      "validation_interval", "validation_cases", "max_train_batches"):
            if field in recorded and getattr(args, field, None) != recorded[field]:
                raise ValueError(f"PSRT resume differs for {field}")
        if not isinstance(state.get("rng"), dict) or not isinstance(state.get("optimizer"), dict):
            raise ValueError("PSRT exact resume requires saved RNG and optimizer states")
        optimizer.load_state_dict(state["optimizer"])
    return state


@torch.no_grad()
def validate(model, loader, p0, device, args):
    model.eval()
    registered = MetricAverager()
    warped = MetricAverager()
    geometry_gen = make_generator(device, args.seed + 60000)
    for batch in loader:
        gt = batch["gt"].to(device, non_blocking=True)
        msi = batch["hr_msi"].to(device, non_blocking=True)
        lr = p0.degrade(gt)
        registered.update(calc_metrics(model(lr, msi), gt, args.scale_ratio))
        if args.stage == "mixed":
            for _ in range(args.validation_cases):
                geometry = sample_synthetic_geometry(
                    *gt.shape[-2:], device=device, dtype=gt.dtype,
                    generator=geometry_gen,
                    local_strength_min_fraction=0.0,
                    local_strength_max_fraction=1.0,
                    **geometry_kwargs(args),
                )
                lr_warp = make_deformed_lr_hsi(gt, geometry, p0)
                warped.update(calc_metrics(model(lr_warp, msi), gt, args.scale_ratio))
    return registered.average(), warped.average()


def append_csv(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(record))
        if fp.tell() == 0:
            writer.writeheader()
        writer.writerow(record)


def main(stage="registered"):
    if stage not in ("registered", "mixed"):
        raise ValueError("PSRT stage must be registered or mixed")
    args = parse_args(stage)
    args.stage = stage
    args.validation_interval = args.validation_interval or DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs, args.batch_size, args.validation_interval, args.validation_cases) < 1:
        raise ValueError("Epochs, batch, validation interval and cases must be positive")
    if args.max_train_batches < 0 or args.grad_clip < 0 or args.lr <= 0 or args.weight_decay < 0:
        raise ValueError("Invalid PSRT optimization settings")
    if args.degradation_mode != "physical":
        raise ValueError("PSRT shared protocol requires physical degradation")
    if stage == "registered" and args.init_checkpoint:
        raise ValueError("Use --resume for stage-1 training rather than --init_checkpoint")
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_set, val_set, _, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(train_set) or not len(val_set):
        raise ValueError("Empty PSRT training/validation split")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    p0 = train_set.degradation_operator.to(device)
    fingerprint = sensor_fingerprint(info)
    model = build_model(info, args, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    family = "physical" if stage == "registered" else "hsi_warp_final"
    ckpt_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else (
        HERE / "checkpoints" / family / args.dataset
    )
    logs_dir = Path(args.log_dir) if args.log_dir else (
        HERE / "logs" / family / args.dataset
    )
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)

    geometry_gen = make_generator(device, args.seed + 404)
    first_epoch, best_psnr, bad_validations = 0, -float("inf"), 0
    if args.resume:
        state = load_checkpoint(args.resume, model, args, expected_stage=stage,
                                fingerprint=fingerprint, optimizer=optimizer)
        first_epoch = state["epoch"] + 1
        best_psnr = float(state["best_psnr"])
        bad_validations = int(state.get("bad_validations", 0))
        rng_restore(state["rng"], geometry_gen)
    elif stage == "mixed":
        init_path = args.init_checkpoint or str(
            HERE / "checkpoints" / "physical" / args.dataset / "best.pth.tar"
        )
        load_checkpoint(init_path, model, args, expected_stage="registered",
                        fingerprint=fingerprint)
        print(f"PSRT mixed initialized from stage-1 model weights: {init_path}", flush=True)

    protocol = {
        "method": "PSRT", "architecture_id": ARCHITECTURE_ID,
        "upstream_source_sha": UPSTREAM_SOURCE_SHA,
        "dataset": args.dataset, "stage": stage,
        "hsi_channels": model.hsi_channels, "msi_channels": model.msi_channels,
        "sensor_fingerprint": fingerprint,
        "loss": "L1", "optimizer": "AdamW", "args": vars(args),
    }
    (ckpt_dir / "run_protocol.txt").write_text(
        json.dumps(protocol, indent=2, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(protocol, indent=2), flush=True)

    for epoch in range(first_epoch, args.epochs):
        model.train()
        total_loss, n_samples, identity_samples = 0.0, 0, 0
        progress = tqdm(train_loader, desc=f"PSRT {stage} {epoch+1}/{args.epochs}",
                        dynamic_ncols=True, mininterval=1.0)
        for batch_index, batch in enumerate(progress):
            if args.max_train_batches and batch_index >= args.max_train_batches:
                break
            gt = batch["gt"].to(device, non_blocking=True)
            msi = batch["hr_msi"].to(device, non_blocking=True)
            with torch.no_grad():
                lr, identities = deformed_observation(
                    gt, p0, args, geometry_gen, mixed=(stage == "mixed")
                )
            optimizer.zero_grad(set_to_none=True)
            pred = model(lr, msi)
            loss = F.l1_loss(pred, gt)
            if not torch.isfinite(loss):
                raise FloatingPointError(
                    f"Nonfinite PSRT L1 loss at epoch={epoch+1}, batch={batch_index}"
                )
            loss.backward()
            if args.grad_clip:
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                if not torch.isfinite(norm):
                    raise FloatingPointError(
                        f"Nonfinite PSRT gradient at epoch={epoch+1}, batch={batch_index}"
                    )
            optimizer.step()
            size = int(gt.shape[0])
            total_loss += float(loss.detach().item()) * size
            identity_samples += identities
            n_samples += size
            progress.set_postfix(L1=f"{total_loss/max(n_samples,1):.6f}", refresh=False)

        train_l1 = total_loss / max(n_samples, 1)
        print(f"PSRT epoch={epoch+1} stage={stage} train_L1={train_l1:.6f} "
              f"identity_fraction={identity_samples/max(n_samples,1):.4f}", flush=True)
        evaluate_now = ((epoch + 1) % args.validation_interval == 0
                        or epoch + 1 == args.epochs)
        if evaluate_now:
            registered, warped = validate(model, val_loader, p0, device, args)
            monitored = warped if stage == "mixed" else registered
            psnr = float(monitored["PSNR"])
            if not np.isfinite(psnr):
                raise FloatingPointError("Nonfinite PSRT validation PSNR")
            improved = psnr > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr, bad_validations = psnr, 0
            else:
                bad_validations += 1
            record = {
                "epoch": epoch + 1, "train_L1": train_l1,
                "identity_fraction": identity_samples / max(n_samples, 1),
                **{f"reg_{key}": value for key, value in registered.items()},
                **{f"warp_{key}": value for key, value in warped.items()},
                "monitored_PSNR": psnr, "best_PSNR": best_psnr,
            }
            append_csv(logs_dir / "validation_history.csv", record)
            print(f"VALIDATION {json.dumps(record)}", flush=True)
            if improved:
                torch.save(checkpoint_payload(
                    model, optimizer, args, epoch, best_psnr, bad_validations,
                    fingerprint, geometry_gen,
                ), ckpt_dir / "best.pth.tar")
            if args.early_stop_patience > 0 and bad_validations >= args.early_stop_patience:
                print(f"PSRT early stopping at epoch={epoch+1}", flush=True)
                torch.save(checkpoint_payload(
                    model, optimizer, args, epoch, best_psnr, bad_validations,
                    fingerprint, geometry_gen,
                ), ckpt_dir / "last.pth.tar")
                break

        torch.save(checkpoint_payload(
            model, optimizer, args, epoch, best_psnr, bad_validations,
            fingerprint, geometry_gen,
        ), ckpt_dir / "last.pth.tar")


if __name__ == "__main__":
    main()
