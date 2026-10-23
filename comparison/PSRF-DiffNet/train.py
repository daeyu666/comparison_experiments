"""PSRF-DiffNet training adapter for the shared comparison protocol.

The active released path is CRN -> FRN -> complementary fusion inside an
x0-prediction diffusion wrapper. Registered and mixed stages intentionally use
the same data split, physical degradation, optimizer family, and validation
schedule as EMR-Diff and CLSNet.
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
    add_data_args,
    add_geometry_args,
    build_shared_cfg,
    deformed_observation,
    geometry_kwargs,
    make_generator,
    resolve_device,
    sample_synthetic_geometry,
    set_seed,
    make_deformed_lr_hsi,
    require_srf_weights,
)
from data_loader import build_datasets  # noqa: E402
from experiment_protocol import DATASET_VALIDATION_INTERVAL  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from autograd_safe import make_autograd_safe  # noqa: E402
from diffusion import PSRFDiffusion  # noqa: E402


ARCHITECTURE_ID = "PSRF-DiffNet-official-active-path-v5"
DATASETS = ["PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg"]


def parser(stage: str = "registered"):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.set_defaults(stage=stage)
    # Kept for compatibility with the root unified launcher. The stage selector
    # controls whether the observation is registered or geometrically deformed.
    p.add_argument("--train_misalignment_mode", choices=["registered"], default="registered")
    p.add_argument("--epochs", type=int, default=800 if stage == "registered" else 600)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--seed", type=int, default=1001 if stage == "registered" else 10)
    p.add_argument("--validation_interval", type=int, default=None)
    p.add_argument("--validation_cases", type=int, default=5)
    p.add_argument("--early_stop_patience", type=int, default=999999)
    p.add_argument("--early_stop_min_delta", type=float, default=0.02)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--val_sample_steps", type=int, default=50)
    p.add_argument("--n_timestep", type=int, default=2000)
    p.add_argument("--linear_start", type=float, default=1e-4)
    p.add_argument("--linear_end", type=float, default=2e-3)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--save_interval", type=int, default=20)
    p.add_argument("--checkpoint_dir", default="")
    p.add_argument("--log_dir", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--init_checkpoint", default="")
    return p


def _snapshot_rng(geometry_gen, diffusion_gen):
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "geometry": geometry_gen.get_state(),
        "diffusion": diffusion_gen.get_state(),
    }


def _restore_rng(state, geometry_gen, diffusion_gen):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda"]])
    if state.get("geometry") is not None:
        geometry_gen.set_state(state["geometry"].cpu())
    if state.get("diffusion") is not None:
        diffusion_gen.set_state(state["diffusion"].cpu())


def checkpoint_payload(model, optimizer, epoch, best, bad, args, geometry_gen, diffusion_gen):
    return {
        "method": "PSRF-DiffNet",
        "architecture_id": ARCHITECTURE_ID,
        "hsi_channels": model.hsi_channels,
        "msi_channels": model.msi_channels,
        "patch_size": model.patch_size,
        "scale_ratio": model.scale_ratio,
        "n_timestep": model.n_timestep,
        "sensor_fingerprint": getattr(model, "sensor_fingerprint", None),
        "training_stage": args.stage,
        "registered_probability": 1.0 if args.stage == "registered" else args.registered_probability,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_psnr": float(best),
        "bad_validations": int(bad),
        "args": vars(args).copy(),
        "rng": _snapshot_rng(geometry_gen, diffusion_gen),
    }


def load_checkpoint(path, model, args, *, expected_stage, optimizer=None):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"PSRF-DiffNet checkpoint not found: {path}")
    state = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    if state.get("method") != "PSRF-DiffNet":
        raise ValueError("Checkpoint is not a PSRF-DiffNet checkpoint")
    if state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError(
            "Checkpoint architecture is stale/incompatible; retrain after the "
            "official-structure alignment update."
        )
    if state.get("training_stage") != expected_stage:
        raise ValueError(
            f"Expected {expected_stage} checkpoint, got {state.get('training_stage')!r}"
        )
    for key, actual in (
        ("hsi_channels", model.hsi_channels),
        ("msi_channels", model.msi_channels),
        ("patch_size", model.patch_size),
        ("scale_ratio", model.scale_ratio),
        ("n_timestep", model.n_timestep),
    ):
        if state.get(key) != actual:
            raise ValueError(f"Checkpoint/model mismatch for {key}: {state.get(key)} != {actual}")
    fingerprint = getattr(model, "sensor_fingerprint", None)
    if fingerprint is not None and state.get("sensor_fingerprint") != fingerprint:
        raise ValueError("Checkpoint sensor SRF fingerprint does not match the current dataset protocol")
    recorded = state.get("args", {})
    for key in (
        "dataset", "image_size", "patch_size", "stride", "scale_ratio",
        "degradation_mode", "mtf_nyquist", "psf_truncate",
    ):
        if key in recorded and recorded[key] != getattr(args, key):
            raise ValueError(
                f"Checkpoint protocol mismatch for {key}: "
                f"{recorded[key]!r} != {getattr(args, key)!r}"
            )
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        if state.get("optimizer") is None or state.get("rng") is None:
            raise ValueError("Exact resume requires optimizer and RNG states")
        for key in ("lr", "weight_decay", "batch_size", "seed", "validation_interval"):
            if recorded.get(key) != getattr(args, key):
                raise ValueError(f"Resume schedule mismatch for {key}")
        optimizer.load_state_dict(state["optimizer"])
    return state


@torch.no_grad()
def evaluate(model, loader, device, p0, args):
    model.eval()
    registered, warped = MetricAverager(), MetricAverager()
    cuda_devices = []
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        cuda_devices = [idx]

    # The guided-filter PCA uses randomized low-rank decomposition. Fork and
    # reset RNG state so validation predictions are identical across epochs
    # without perturbing the training RNG stream.
    with torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        if cuda_devices:
            torch.cuda.manual_seed_all(args.eval_seed)
        for batch in loader:
            geometry_gen = make_generator(device, args.seed + 60000)
            sample_gen = torch.Generator(device="cpu").manual_seed(args.eval_seed)
            gt = batch["gt"].to(device, non_blocking=True)
            msi = batch["hr_msi"].to(device, non_blocking=True)
            lr_registered = p0.degrade(gt)
            pred, _ = model.sample_tiled(
                lr_registered, msi, sample_steps=args.val_sample_steps,
                tile_stride=args.patch_size, generator=sample_gen,
            )
            registered.update(calc_metrics(pred, gt, args.scale_ratio))

            if args.stage == "mixed":
                for _ in range(args.validation_cases):
                    geometry = sample_synthetic_geometry(
                        *gt.shape[-2:], device=device, dtype=gt.dtype,
                        generator=geometry_gen, local_strength_min_fraction=0.0,
                        local_strength_max_fraction=1.0, **geometry_kwargs(args),
                    )
                    lr_warped = make_deformed_lr_hsi(gt, geometry, p0)
                    pred, _ = model.sample_tiled(
                        lr_warped, msi, sample_steps=args.val_sample_steps,
                        tile_stride=args.patch_size, generator=sample_gen,
                    )
                    warped.update(calc_metrics(pred, gt, args.scale_ratio))
    return registered.average(), warped.average()


def main(stage: str = "registered"):
    if stage not in ("registered", "mixed"):
        raise ValueError(f"Unsupported PSRF stage: {stage}")
    args = parser(stage).parse_args()
    args.stage = stage
    args.validation_interval = args.validation_interval or DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs, args.batch_size, args.validation_interval, args.validation_cases, args.val_sample_steps) < 1:
        raise ValueError("Epochs, batch size, validation interval/cases and sampling steps must be positive")
    if stage == "mixed" and args.degradation_mode != "physical":
        raise ValueError("The unified mixed-stage protocol requires physical degradation")
    if stage == "registered" and args.init_checkpoint:
        raise ValueError("Registered training starts from scratch; use --resume to continue")
    set_seed(args.seed)
    device = resolve_device(args.device)
    train_set, val_set, _, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if len(train_set) == 0 or len(val_set) == 0:
        raise ValueError("PSRF-DiffNet requires nonempty train and validation splits")
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    p0 = train_set.degradation_operator.to(device)

    model = PSRFDiffusion(
        patch_size=args.patch_size,
        scale_ratio=args.scale_ratio,
        hsi_channels=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        n_timestep=args.n_timestep,
        linear_start=args.linear_start,
        linear_end=args.linear_end,
    ).to(device)
    model.sensor_fingerprint = hashlib.sha256(
        require_srf_weights(info).detach().cpu().numpy().tobytes()
    ).hexdigest()
    make_autograd_safe(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    family = "physical" if stage == "registered" else "hsi_warp_final"
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else THIS_DIR / "checkpoints" / family / args.dataset
    log_dir = Path(args.log_dir) if args.log_dir else THIS_DIR / "logs" / family / args.dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    geometry_gen = make_generator(device, args.seed + 404)
    diffusion_gen = torch.Generator(device="cpu").manual_seed(args.seed + 15485863)
    start_epoch, best_psnr, bad_validations = 0, -float("inf"), 0

    if args.resume:
        state = load_checkpoint(args.resume, model, args, expected_stage=stage, optimizer=optimizer)
        start_epoch = int(state["epoch"]) + 1
        best_psnr = float(state["best_psnr"])
        bad_validations = int(state.get("bad_validations", 0))
        _restore_rng(state["rng"], geometry_gen, diffusion_gen)
    elif stage == "mixed":
        init_path = args.init_checkpoint or str(THIS_DIR / "checkpoints" / "physical" / args.dataset / "best.pth.tar")
        state = load_checkpoint(init_path, model, args, expected_stage="registered")
        # Only model weights transfer between stages; AdamW state is new.
        print(f"Loaded registered PSRF-DiffNet weights; reinitialized AdamW: {init_path}")

    protocol = {
        "method": "PSRF-DiffNet",
        "architecture_id": ARCHITECTURE_ID,
        "dataset": args.dataset,
        "hsi_channels": model.hsi_channels,
        "msi_channels": model.msi_channels,
        "scale_ratio": args.scale_ratio,
        "degradation_mode": args.degradation_mode,
        "train_patch": args.patch_size,
        "train_stride": args.stride,
        "validation_interval": args.validation_interval,
        "validation_cases": args.validation_cases,
        "monitor": "registered_PSNR" if stage == "registered" else "warp_PSNR",
        "optimizer": "Adam",
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "stage": stage,
        "registered_probability": 1.0 if stage == "registered" else args.registered_probability,
        "arguments": vars(args),
    }
    (checkpoint_dir / "run_protocol.txt").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)

    history = log_dir / "validation_history.csv"
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"PSRF-DiffNet: dataset={args.dataset}, HSI={model.hsi_channels}, MSI={model.msi_channels}, "
          f"scale=x{args.scale_ratio}, params={params:,}, stage={stage}", flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        loss_sum, count, identities = 0.0, 0, 0
        progress = tqdm(
            train_loader, desc=f"PSRF-DiffNet {stage} {epoch + 1}/{args.epochs}",
            dynamic_ncols=True, mininterval=1.0,
        )
        for batch in progress:
            gt = batch["gt"].to(device, non_blocking=True)
            msi = batch["hr_msi"].to(device, non_blocking=True)
            with torch.no_grad():
                lr_hsi, identity_count = deformed_observation(
                    gt, p0, args, geometry_gen, mixed=(stage == "mixed")
                )
            optimizer.zero_grad(set_to_none=True)
            out = model(gt, lr_hsi, msi, generator=diffusion_gen)
            if not bool(torch.isfinite(out.loss)):
                raise FloatingPointError(f"PSRF-DiffNet nonfinite loss at epoch {epoch + 1}")
            out.loss.backward()
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            bs = int(gt.shape[0])
            count += bs
            identities += identity_count
            loss_sum += float(out.loss.item()) * bs
            progress.set_postfix(
                l1=f"{loss_sum / max(count, 1):.6f}",
                identity=f"{identities / max(count, 1):.3f}",
                refresh=False,
            )

        mean_loss = loss_sum / max(count, 1)
        identity_fraction = identities / max(count, 1)
        print(f"PSRF-DiffNet {stage} {epoch + 1}/{args.epochs} "
              f"loss={mean_loss:.6f} identity={identity_fraction:.3f}", flush=True)

        do_eval = (epoch + 1) % args.validation_interval == 0 or epoch + 1 == args.epochs
        if do_eval:
            reg, warp = evaluate(model, val_loader, device, p0, args)
            monitored = warp if stage == "mixed" else reg
            score = float(monitored["PSNR"])
            improved = score > best_psnr + args.early_stop_min_delta
            if improved:
                best_psnr, bad_validations = score, 0
            else:
                bad_validations += 1
            row = {
                "epoch": epoch + 1,
                "train_l1": mean_loss,
                "identity_fraction": identity_fraction,
                **{f"reg_{k}": v for k, v in reg.items()},
                **{f"warp_{k}": v for k, v in warp.items()},
                "monitored_PSNR": score,
                "best_PSNR": best_psnr,
            }
            with history.open("a", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
            print(f"VALIDATION {row}", flush=True)
            if improved:
                torch.save(
                    checkpoint_payload(model, optimizer, epoch, best_psnr, bad_validations,
                                       args, geometry_gen, diffusion_gen),
                    checkpoint_dir / "best.pth.tar",
                )
            if args.early_stop_patience > 0 and bad_validations >= args.early_stop_patience:
                print("Validation early stopping", flush=True)
                torch.save(
                    checkpoint_payload(model, optimizer, epoch, best_psnr, bad_validations,
                                       args, geometry_gen, diffusion_gen),
                    checkpoint_dir / "last.pth.tar",
                )
                break

        # Save exact continuation state after every epoch so the root
        # launcher can resume without silently losing recent optimizer/RNG state.
        torch.save(
            checkpoint_payload(model, optimizer, epoch, best_psnr, bad_validations,
                               args, geometry_gen, diffusion_gen),
            checkpoint_dir / "last.pth.tar",
        )


if __name__ == "__main__":
    main()
