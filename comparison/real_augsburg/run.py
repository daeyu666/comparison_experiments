"""Train new real Augsburg adapters or reconstruct/score the common native ROI."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison.UAFL.augsburg2_wald_common import (
    WaldDataset, correct_msi, metrics_from_sums, metrics_sums,
)
from comparison.UAFL.augsburg2_wald_qnr import evaluate_cache
from comparison.UAFL.common import set_seed
from comparison.real_augsburg.common import WaldCache, chw, positions, write_geotiff, write_json
from comparison.real_augsburg.methods import architecture_id, build_model, classical_solver, training_loss
from comparison.registered_fusion.runner import restore_rng, save_state, snapshot_rng
from experiment_protocol import (
    REAL_CLASSICAL_METHODS, REAL_EVAL_INTERVAL, REAL_EVAL_PATCH, REAL_EVAL_SEED,
    REAL_EPOCHS, REAL_LR, REAL_MIN_VALID_FRACTION, REAL_MONITOR, REAL_NEW_NEURAL_METHODS,
    REAL_QNR_MIN_VALID_FRACTION, REAL_QNR_SUPPORT_FRACTION, REAL_QNR_WINDOW_HR,
    REAL_RADIOMETRY, REAL_SAVE_INTERVAL, REAL_SEED, REAL_TILE_SIZE, REAL_TILE_STRIDE,
    REAL_TRAIN_PATCH, REAL_TRAIN_STRIDE, REAL_WALD_ROOT, REAL_WEIGHT_DECAY,
    real_checkpoint_dir, real_log_dir, real_output_dir, resolve_model,
)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", choices=(*REAL_CLASSICAL_METHODS, *REAL_NEW_NEURAL_METHODS), required=True)
    p.add_argument("--stage", choices=("train", "test", "native"), required=True)
    p.add_argument("--wald_root", default=ROOT / REAL_WALD_ROOT, type=Path)
    p.add_argument("--radiometry_json", default=ROOT / REAL_RADIOMETRY, type=Path)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--checkpoint_dir", type=Path)
    p.add_argument("--log_dir", type=Path)
    p.add_argument("--save_root", type=Path)
    p.add_argument("--resume", type=Path)
    p.add_argument("--center_holdout", action="store_true")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--write_tif", action="store_true")
    for name, default in (
        ("train_patch_size", REAL_TRAIN_PATCH), ("train_stride", REAL_TRAIN_STRIDE),
        ("eval_patch_size", REAL_EVAL_PATCH), ("min_valid_fraction", REAL_MIN_VALID_FRACTION),
    ):
        p.add_argument("--" + name, type=type(default), default=default, choices=(default,))
    for name, default in (
        ("epochs", REAL_EPOCHS), ("batch_size", 1), ("num_workers", 0),
        ("lr", REAL_LR), ("weight_decay", REAL_WEIGHT_DECAY),
        ("eval_interval", REAL_EVAL_INTERVAL), ("save_interval", REAL_SAVE_INTERVAL),
        ("seed", REAL_SEED), ("eval_seed", REAL_EVAL_SEED),
        ("tile_size", REAL_TILE_SIZE), ("tile_stride", REAL_TILE_STRIDE),
        ("qnr_window_hr", REAL_QNR_WINDOW_HR), ("qnr_min_valid_fraction", REAL_QNR_MIN_VALID_FRACTION),
        ("qnr_support_fraction", REAL_QNR_SUPPORT_FRACTION),
    ):
        p.add_argument("--" + name, type=type(default), default=default)
    p.add_argument("--monitor", default=REAL_MONITOR, choices=("ref_sam", "ref_psnr"))
    return p


def device_for(args):
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable; use --device cpu for a smoke check")
    return device


def model_metadata(method, cache):
    return {**cache.fingerprint, "method": method, "architecture_id": architecture_id(method),
            "protocol": "Augsburg-2-Wald-Comparison-v1"}


def load_checkpoint(path, model, metadata):
    state = torch.load(path, map_location="cpu", weights_only=True)
    for k, v in metadata.items():
        if state.get("metadata", {}).get(k) != v:
            raise ValueError(f"{path}: Wald checkpoint differs for {k}; use weights trained on this split")
    model.load_state_dict(state["model"], strict=True)
    return state


def pack(batch, device, cache):
    lr, msi, gt, mask = (batch[key].to(device) for key in ("lr_hsi", "hr_msi", "gt", "valid_mask"))
    return lr, correct_msi(msi, cache.calibration), gt, mask


def loader(cache, split, args):
    data = WaldDataset(cache.root, split, args.train_patch_size, args.train_stride,
                       args.eval_patch_size, args.min_valid_fraction)
    train = split == "train"
    return DataLoader(data, batch_size=args.batch_size if train else 1,
                      shuffle=train, num_workers=args.num_workers if train else 0)


@torch.no_grad()
def reference_metrics(predict, batches, cache, device):
    values = [0, 0, 0, 0]
    for batch in batches:
        lr, msi, gt, mask = pack(batch, device, cache)
        for i, v in enumerate(metrics_sums(predict(lr, msi), gt, mask)):
            values[i] += v
    result = metrics_from_sums(values)
    if not all(math.isfinite(v) for v in result.values()):
        raise FloatingPointError(f"Nonfinite Wald reference scores: {result}")
    return result


def train(args, cache, adapter):
    if args.model in REAL_CLASSICAL_METHODS:
        raise ValueError(f"{args.model} is a per-pair solver and does not train")
    if min(args.epochs, args.batch_size, args.eval_interval, args.save_interval) < 1:
        raise ValueError("epochs, batch and intervals must be positive")
    if not math.isfinite(args.lr) or args.lr <= 0 or args.weight_decay < 0 or args.num_workers < 0:
        raise ValueError("Invalid optimizer or loader options")
    paths = args.checkpoint_dir or real_checkpoint_dir(adapter)
    logs = args.log_dir or real_log_dir(adapter)
    history = logs / "history.csv"
    if not args.resume and ((paths / "last.pth.tar").exists() or (paths / "best.pth.tar").exists() or history.exists()):
        raise FileExistsError("Existing Wald run: use --resume auto or distinct checkpoint/log directories")
    device = device_for(args)
    set_seed(args.seed)
    train_set, val_set = (loader(cache, split, args) for split in ("train", "validation"))
    model = build_model(args.model, device)
    loss_fn = training_loss(args.model)
    metadata = model_metadata(args.model, cache)
    settings = {k: getattr(args, k) for k in (
        "lr", "weight_decay", "batch_size", "seed", "num_workers", "monitor", "eval_interval",
    )}
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start = 0
    best = math.inf if args.monitor == "ref_sam" else -math.inf
    if args.resume:
        state = load_checkpoint(args.resume, model, metadata)
        if state.get("settings") != settings:
            raise ValueError("Resume requires the original Wald optimizer/selection settings")
        optimizer.load_state_dict(state["optimizer"])
        restore_rng(state["rng"], device)
        start, best = int(state["epoch"]), float(state["best"])
    paths.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    write_json(paths / "run_protocol.json", {**metadata, **settings,
               "train_tiles": len(train_set.dataset), "validation_tiles": len(val_set.dataset)})
    clip = {"PSRT": 10.0, "PMI-RFCoNet": 1.0}.get(args.model)
    for epoch in range(start + 1, args.epochs + 1):
        model.train()
        total = count = 0
        for batch_idx, batch in enumerate(train_set, 1):
            lr, msi, gt, mask = pack(batch, device, cache)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model, lr, msi, gt, mask)
            if not torch.isfinite(loss):
                raise FloatingPointError(f"{args.model}: nonfinite loss at epoch {epoch}, batch {batch_idx}")
            loss.backward()
            if clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(gt)
            count += len(gt)
        row = {"epoch": epoch, "loss": total / count}
        evaluate = epoch % args.eval_interval == 0 or epoch == args.epochs
        improved = False
        if evaluate:
            model.eval()
            scores = reference_metrics(model, val_set, cache, device)
            score = scores[args.monitor]
            improved = score < best if args.monitor == "ref_sam" else score > best
            if improved:
                best = score
            row.update(scores, best=best)
            with history.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=list(row))
                if f.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        print(f"WALD_TRAIN {args.model} {row}", flush=True)
        if improved or evaluate or epoch % args.save_interval == 0:
            state = {"metadata": metadata, "settings": settings, "epoch": epoch, "best": best,
                     "model": model.state_dict(), "optimizer": optimizer.state_dict(), "rng": snapshot_rng(device)}
            if improved:
                save_state(state, paths / "best.pth.tar")
            save_state(state, paths / "last.pth.tar")


@torch.no_grad()
def tiled_predict(model, lr, msi, calibration, device, tile, stride):
    h, w = msi.shape[:2]
    result = np.zeros((h, w, lr.shape[-1]), np.float32)
    weights = np.zeros((h, w), np.float32)
    for y in positions(h, tile, stride):
        for x in positions(w, tile, stride):
            obs = chw(lr[y // 3:(y + tile) // 3, x // 3:(x + tile) // 3], device)
            guide = chw(msi[y:y + tile, x:x + tile], device)
            tile_pred = model(obs, correct_msi(guide, calibration))
            arr = tile_pred[0].permute(1, 2, 0).float().cpu().numpy()
            if arr.shape != (tile, tile, lr.shape[-1]) or not np.isfinite(arr).all():
                raise FloatingPointError(f"Invalid native prediction shape/values at tile {(y, x)}")
            result[y:y + tile, x:x + tile] += arr
            weights[y:y + tile, x:x + tile] += 1
    return result / weights[..., None]


def evaluate(args, cache, adapter):
    classical = args.model in REAL_CLASSICAL_METHODS
    dest = args.save_root or real_output_dir(adapter)
    if classical:
        device = torch.device("cpu")
        solve = classical_solver(args.model, cache)
        checkpoint = None
        metadata = {**cache.fingerprint, "method": args.model}

        def model(lr, msi):
            arr, _ = solve(lr[0].permute(1, 2, 0).numpy(), msi[0].permute(1, 2, 0).numpy())
            return chw(arr, device)
    else:
        device = device_for(args)
        checkpoint = args.checkpoint or real_checkpoint_dir(adapter) / "best.pth.tar"
        metadata = model_metadata(args.model, cache)
        model = build_model(args.model, device).eval()
        state = load_checkpoint(checkpoint, model, metadata)
        metadata["checkpoint_epoch"] = state["epoch"]
        del state
    set_seed(args.eval_seed)
    if args.stage == "test":
        scores = reference_metrics(model, loader(cache, "test", args), cache, device)
        write_json(dest / "reference.json", {**metadata, **scores})
        print(f"WALD_REFERENCE {args.model} {scores}")
        return
    lr, msi, mask, geo = cache.native_inputs()
    before = time.perf_counter()
    if classical:
        gain, bias = cache.calibration
        cube, solver_report = solve(lr, msi * gain + bias)
    else:
        if any(n % 24 for n in (args.tile_size, args.tile_stride)):
            raise ValueError("Wald native tile and stride must be multiples of 24")
        cube = tiled_predict(model, lr, msi, cache.calibration, device, args.tile_size, args.tile_stride)
        solver_report = None
    seconds = time.perf_counter() - before
    if cube.shape != (*mask.shape, 242) or not np.isfinite(cube).all():
        raise FloatingPointError("Native reconstruction has incorrect dimensions or nonfinite values")
    cube[~mask.astype(bool)] = 0
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "fused_hsi.npy"
    np.save(output, cube)
    if args.write_tif:
        write_geotiff(dest / "fused_hsi.tif", cube, geo)
    scores = evaluate_cache(
        cache.root, output, cache.radiometry, window_hr=args.qnr_window_hr,
        min_valid_fraction=args.qnr_min_valid_fraction, support_fraction=args.qnr_support_fraction,
    )
    write_json(dest / "qnr.json", scores)
    write_json(dest / "protocol.json", {
        **metadata, "checkpoint": str(checkpoint.resolve()) if checkpoint else None,
        "device": str(device), "output_shape": list(cube.shape), "inference_seconds": seconds,
        "tile_size": None if classical else args.tile_size, "tile_stride": None if classical else args.tile_stride,
        "solver": solver_report, "eval_seed": args.eval_seed,
    })
    print(f"WALD_QNR {args.model} QNR={scores['QNR']:.6f} Dlambda={scores['Dlambda']:.6f} Ds={scores['Ds']:.6f}")


def main():
    args = parser().parse_args()
    adapter = resolve_model(args.model)
    cache = WaldCache.load(args.wald_root, args.radiometry_json)
    if args.stage == "train":
        train(args, cache, adapter)
    else:
        evaluate(args, cache, adapter)


if __name__ == "__main__":
    main()
