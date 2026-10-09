"""Train/test EMR-Diff on Augsburg-2 center-heldout strict Wald x3.

This runner intentionally accepts only the new leakage-controlled central
holdout cache. It reuses UAFL's exact WaldDataset and pooled masked metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from wald_emr_common import (
    WaldDataset, PROTOCOL, PROVENANCE, build_diffusion, build_model,
    metrics_from_sums, metrics_sums, pack_batch, predict_batch,
    read_json, read_radiometry, require_center_holdout,
    training_step, verify_checkpoint,
)
from EMRDiff import Edge


def parse_args():
    p = argparse.ArgumentParser(
        description="EMR-Diff Augsburg-2 center-heldout strict Wald x3"
    )
    p.add_argument("--stage", choices=("train", "test"), default="train")
    p.add_argument(
        "--wald_root",
        default="../S2Diff-MH/data/augsburg2_wald_center_holdout",
    )
    p.add_argument(
        "--radiometry_json",
        default="../S2Diff-MH/data/calibration/Augsburg2_Wald_center_holdout_radiometry.json",
    )
    p.add_argument(
        "--checkpoint_dir",
        default="./comparison/EMR-Diff/checkpoints/augsburg2_wald_center_holdout",
    )
    p.add_argument(
        "--log_dir",
        default="./comparison/EMR-Diff/logs/augsburg2_wald_center_holdout",
    )
    p.add_argument("--checkpoint", default="")
    p.add_argument("--resume", default="")
    p.add_argument("--train_patch_size", type=int, default=24)
    p.add_argument("--train_stride", type=int, default=6)
    p.add_argument("--eval_patch_size", type=int, default=48)
    p.add_argument("--min_valid_fraction", type=float, default=0.80)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=5e-5)
    p.add_argument("--eval_interval", type=int, default=5)
    p.add_argument("--save_interval", type=int, default=5)
    p.add_argument("--monitor", choices=("ref_sam", "ref_psnr"), default="ref_sam")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--model_width", type=int, default=64,
        help="BAFUNet latent width; diffusion state remains 242+4=246 channels",
    )
    return p.parse_args()


def rng_snapshot():
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "numpy": np.random.get_state(),
        "python": random.getstate(),
    }


def restore_rng(rng):
    if rng is None:
        return
    torch.set_rng_state(rng["torch"].cpu())
    if torch.cuda.is_available() and rng.get("cuda") is not None:
        torch.cuda.set_rng_state_all([state.cpu() for state in rng["cuda"]])
    np.random.set_state(rng["numpy"])
    random.setstate(rng["python"])


def save_state(
    path, *, model, optimizer, epoch, best, args, sha, sigma,
    split_protocol_id, test_bbox_30m, forbidden_bbox_30m,
):
    data = {
        "protocol": PROTOCOL,
        "msi_source": PROVENANCE,
        "target": "observed_30m_HSI_only",
        "scale_ratio": 3,
        "train_area": read_json(
            Path(args.wald_root) / "train" / "meta.json"
        )["training_area"],
        "split_protocol_id": split_protocol_id,
        "test_bbox_30m": list(map(int, test_bbox_30m)),
        "forbidden_bbox_30m": list(map(int, forbidden_bbox_30m)),
        "wald_sigma": float(sigma),
        "radiometry_sha256": sha,
        "model_width": args.model_width,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": int(epoch),
        "best_metric": float(best),
        "monitor": args.monitor,
        "args": vars(args),
        "rng": rng_snapshot(),
    }
    torch.save(data, path)


def load_state(
    path, model, *, sha, sigma, device, split_protocol_id, test_bbox_30m,
    forbidden_bbox_30m, optimizer=None, monitor=None,
):
    state = torch.load(path, map_location=device, weights_only=False)
    verify_checkpoint(
        state,
        radiometry_sha=sha,
        sigma=sigma,
        split_protocol_id=split_protocol_id,
        test_bbox_30m=test_bbox_30m,
        forbidden_bbox_30m=forbidden_bbox_30m,
        monitor=monitor,
        width=model.width,
    )
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None:
        if state.get("optimizer_state_dict") is None:
            raise ValueError("Cannot resume without matching optimizer state")
        optimizer.load_state_dict(state["optimizer_state_dict"])
    return state


@torch.no_grad()
def evaluate(model, diffusion, edge, loader, device, calibration, eval_seed):
    model.eval()
    sums = [0.0, 0, 0.0, 0]
    gpu = (
        [device.index if device.index is not None else torch.cuda.current_device()]
        if device.type == "cuda" else []
    )
    with torch.random.fork_rng(devices=gpu):
        torch.manual_seed(eval_seed)
        if gpu:
            torch.cuda.manual_seed_all(eval_seed)
        for batch in loader:
            pred, (gt, mask) = predict_batch(
                model, diffusion, edge, batch, device, calibration
            )
            batch_sums = metrics_sums(pred, gt, mask)
            sums = [x + y for x, y in zip(sums, batch_sums)]
    return metrics_from_sums(sums)


def main():
    args = parse_args()
    if (
        args.epochs < 1 or args.eval_interval < 1 or args.save_interval < 1
        or args.batch_size < 1
    ):
        raise ValueError("epochs / intervals / batch must be positive")
    if (
        args.train_patch_size != 24
        or args.train_stride != 6
        or args.eval_patch_size != 48
    ):
        raise ValueError(
            "Center-heldout protocol is frozen: train_patch=24, "
            "train_stride=6, eval_patch=48"
        )
    if not 0 < args.min_valid_fraction <= 1:
        raise ValueError("min_valid_fraction must be in (0,1]")
    if args.resume and (args.stage != "train" or args.checkpoint):
        raise ValueError("--resume is for training only")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA is unavailable")

    sigma, split_protocol_id, test_bbox_30m, forbidden_bbox_30m = (
        require_center_holdout(args.wald_root)
    )
    radiometry_path = Path(args.radiometry_json)
    sha = hashlib.sha256(radiometry_path.read_bytes()).hexdigest()
    calibration = read_radiometry(radiometry_path, args.wald_root)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device(args.device)
    model = build_model(args.model_width, device=device)
    diffusion = build_diffusion(device)
    edge = Edge().to(device)
    checkpoints = Path(args.checkpoint_dir)
    logs = Path(args.log_dir)
    params = sum(p.numel() for p in model.parameters())

    if args.stage == "test":
        ckpt = Path(args.checkpoint) if args.checkpoint else checkpoints / "best.pth.tar"
        load_state(
            ckpt, model, sha=sha, sigma=sigma, device=device,
            split_protocol_id=split_protocol_id, test_bbox_30m=test_bbox_30m,
            forbidden_bbox_30m=forbidden_bbox_30m, monitor=args.monitor,
        )
        dataset = WaldDataset(
            args.wald_root, "test", args.train_patch_size, args.train_stride,
            args.eval_patch_size, args.min_valid_fraction,
        )
        scores = evaluate(
            model, diffusion, edge,
            DataLoader(dataset, batch_size=1, shuffle=False),
            device, calibration, args.eval_seed,
        )
        print(
            f"EMR_WALD_CENTER_TEST samples={len(dataset)} "
            f"REF_PSNR={scores['ref_psnr']:.6f} "
            f"REF_SAM={scores['ref_sam']:.6f} "
            f"REF_RMSE={scores['ref_rmse']:.8f} "
            f"ROI30={test_bbox_30m}"
        )
        return

    # WaldDataset reads the train metadata's forbidden_bbox_30m and hard-rejects
    # every 24x24 tile intersecting center+guard. Test is never opened here.
    train_set = WaldDataset(
        args.wald_root, "train", args.train_patch_size, args.train_stride,
        args.eval_patch_size, args.min_valid_fraction,
    )
    val_set = WaldDataset(
        args.wald_root, "validation", args.train_patch_size, args.train_stride,
        args.eval_patch_size, args.min_valid_fraction,
    )
    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, drop_last=False,
    )
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    best = float("inf") if args.monitor == "ref_sam" else -float("inf")
    start = 0
    if args.resume:
        state = load_state(
            args.resume, model, sha=sha, sigma=sigma, device=device,
            split_protocol_id=split_protocol_id, test_bbox_30m=test_bbox_30m,
            forbidden_bbox_30m=forbidden_bbox_30m,
            optimizer=optimizer, monitor=args.monitor,
        )
        start = int(state["epoch"]) + 1
        best = float(state["best_metric"])
        restore_rng(state.get("rng"))

    checkpoints.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    history = logs / "history.csv"

    print(
        f"EMR_AUGSBURG2_CENTER_HOLDOUT train={len(train_set)} "
        f"validation={len(val_set)} patch=24 stride=6 eval_patch=48 "
        f"scale=x3 sigma={sigma} parameters={params} state=246 "
        f"trunk_width={args.model_width} source={PROVENANCE} "
        f"split_protocol_id={split_protocol_id} test_bbox_30m={test_bbox_30m} "
        f"forbidden_bbox_30m={forbidden_bbox_30m} EnMAP10_used=False "
        f"train_augmentation=False monitoring={args.monitor}"
    )

    for epoch in range(start, args.epochs):
        model.train()
        total, count = 0.0, 0
        for batch in train_loader:
            gt, lr_hr, msi, mask, _ = pack_batch(batch, device, calibration)
            optimizer.zero_grad(set_to_none=True)
            loss = training_step(model, diffusion, edge, gt, lr_hr, msi, mask)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(
                    f"EMR center-heldout non-finite loss at epoch {epoch + 1}"
                )
            loss.backward()
            optimizer.step()
            total += float(loss.item()) * gt.shape[0]
            count += gt.shape[0]

        current = epoch + 1
        is_eval = current % args.eval_interval == 0 or current == args.epochs
        msg = (
            f"EMR_WALD_CENTER_EPOCH {current}/{args.epochs} "
            f"LOSS={total / max(count, 1):.7f}"
        )
        if is_eval:
            metrics = evaluate(
                model, diffusion, edge, val_loader,
                device, calibration, args.eval_seed,
            )
            score = metrics[args.monitor]
            if not math.isfinite(score):
                raise FloatingPointError(f"Non-finite validation {args.monitor}")
            improved = score < best if args.monitor == "ref_sam" else score > best
            if improved:
                best = score
                save_state(
                    checkpoints / "best.pth.tar",
                    model=model, optimizer=optimizer, epoch=epoch, best=best,
                    args=args, sha=sha, sigma=sigma,
                    split_protocol_id=split_protocol_id,
                    test_bbox_30m=test_bbox_30m,
                    forbidden_bbox_30m=forbidden_bbox_30m,
                )
            msg += (
                f" VAL_PSNR={metrics['ref_psnr']:.6f}"
                f" VAL_SAM={metrics['ref_sam']:.6f}"
                f" VAL_RMSE={metrics['ref_rmse']:.8f}"
                f" best_{args.monitor}={best:.6f} saved_best={improved}"
            )
            with history.open("a", newline="", encoding="utf-8") as handle:
                row = {
                    "epoch": current,
                    "train_loss": total / max(count, 1),
                    "val_ref_psnr": metrics["ref_psnr"],
                    "val_ref_sam": metrics["ref_sam"],
                    "val_ref_rmse": metrics["ref_rmse"],
                    "best": best,
                }
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if handle.tell() == 0:
                    writer.writeheader()
                writer.writerow(row)
        print(msg)

        if is_eval or current % args.save_interval == 0:
            save_state(
                checkpoints / "last.pth.tar",
                model=model, optimizer=optimizer, epoch=epoch, best=best,
                args=args, sha=sha, sigma=sigma,
                split_protocol_id=split_protocol_id,
                test_bbox_30m=test_bbox_30m,
                forbidden_bbox_30m=forbidden_bbox_30m,
            )

    print(
        f"EMR_WALD_CENTER_TRAIN_COMPLETE best_{args.monitor}={best:.6f} "
        f"best={checkpoints / 'best.pth.tar'}"
    )


if __name__ == "__main__":
    main()
