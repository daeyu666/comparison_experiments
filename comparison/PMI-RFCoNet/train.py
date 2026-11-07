"""Supervised PMI-RFCoNet training under the common registered/mixed x4 protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import shutil
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
if not __package__:
    __package__ = "comparison.PMI-RFCoNet"

from .common import (  # noqa: E402
    add_data_args, add_geometry_args, build_shared_cfg, deformed_observation,
    geometry_kwargs, make_deformed_lr_hsi, make_generator,
    require_srf_weights, resolve_device, sample_synthetic_geometry, set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from experiment_protocol import DATASET_VALIDATION_INTERVAL  # noqa: E402
from .model import ARCHITECTURE_ID, UPSTREAM_COMMIT, UPSTREAM_BLOB_SHA, build_model  # noqa: E402
from .stability import (fail_training, ensure_finite_checkpoint, reconfigure_learning_rate,
                        repair_bn_checkpoint, recalibrate_bn, nonfinite_bn_buffers)


def parser(stage="registered", argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--train_misalignment_mode", choices=["registered"], default="registered")
    p.add_argument("--epochs", type=int, default=800 if stage == "registered" else 600)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight_decay", type=float, default=0.0)
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
    p.add_argument("--resume_optimizer_override", action="store_true",
                   help="Explicitly change lr/grad_clip when resuming; retain Adam moments, schedule phase and RNG")
    p.add_argument("--repair_bn_stats",action="store_true",help="Explicitly reset damaged BN statistics and recalibrate using training observations; requires --resume")
    p.add_argument("--bn_calibration_batches",type=int,default=32)
    p.add_argument("--mid_channels", type=int, default=64)
    p.add_argument("--num_fc", type=int, default=4)
    p.add_argument("--drop", type=float, default=0.0)
    p.add_argument("--lr_milestones", type=int, nargs="+", default=[50,150,300])
    p.add_argument("--lr_gamma", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=0.0)
    p.add_argument("--max_train_batches", type=int, default=0,
                   help="Smoke-test only; 0 means the full training split")
    return p.parse_args(argv)


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


def checkpoint_payload(model, optimizer, scheduler, args, epoch, best_psnr, bad_validations,
                       fingerprint, geometry_gen):
    payload = {
        "method": "PMI-RFCoNet",
        "architecture_id": ARCHITECTURE_ID,
        "upstream_blob_sha": UPSTREAM_BLOB_SHA,
        "upstream_commit": UPSTREAM_COMMIT,
        "epoch": int(epoch),
        "training_stage": args.stage,
        "hsi_channels": model.hsi_channels,
        "msi_channels": model.msi_channels,
        "model_args": model.model_args,
        "registered_probability": 1.0 if args.stage == "registered" else args.registered_probability,
        "sensor_fingerprint": fingerprint,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "args": vars(args).copy(),
        "best_psnr": float(best_psnr),
        "bad_validations": int(bad_validations),
        "bn_stats_repair": getattr(args,"bn_stats_repair_info",None),
        "rng": snapshot_rng(geometry_gen),
    }
    ensure_finite_checkpoint(payload)
    return payload


def load_checkpoint(path, model, args, *, expected_stage, fingerprint, optimizer=None, scheduler=None):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("method") != "PMI-RFCoNet" or state.get("architecture_id") != ARCHITECTURE_ID:
        raise ValueError("Checkpoint is not compatible with this PMI-RFCoNet architecture")
    if state.get("upstream_blob_sha") != UPSTREAM_BLOB_SHA:
        raise ValueError("PMI-RFCoNet upstream source mismatch")
    if state.get("training_stage") != expected_stage:
        raise ValueError(f"Expected {expected_stage} checkpoint, got {state.get('training_stage')}")
    if state.get("sensor_fingerprint") != fingerprint:
        raise ValueError("Checkpoint uses a different sensor SRF")
    if state.get("hsi_channels") != model.hsi_channels or state.get("msi_channels") != model.msi_channels:
        raise ValueError("Checkpoint HSI/MSI channel counts do not match current sensor")
    expected_config = model.model_args
    if state.get("model_args") != expected_config:
        raise ValueError(f"PMI-RFCoNet architecture/inference mismatch: {state.get('model_args')} != {expected_config}")
    recorded = state.get("args", {})
    for name in ("dataset", "image_size", "patch_size", "stride", "scale_ratio",
                 "degradation_mode", "mtf_nyquist", "psf_truncate"):
        if name in recorded and getattr(args, name, None) != recorded[name]:
            raise ValueError(f"PMI-RFCoNet checkpoint protocol mismatch for {name}")
    repaired=None
    if getattr(args,"repair_bn_stats",False):
        repaired=repair_bn_checkpoint(state,model)
    ensure_finite_checkpoint(state)
    state["_bn_recalibration_required"]=repaired
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        for name in ("seed", "lr", "weight_decay", "batch_size",
                     "validation_interval", "validation_cases", "max_train_batches",
                     "lr_milestones", "lr_gamma", "grad_clip", "registered_probability",
                     "max_translation", "max_rotation_deg", "max_local_px", "control_grid", "min_jacobian"):
            if name in recorded and getattr(args, name, None) != recorded[name]:
                if name not in ("lr", "grad_clip") or not getattr(args,"resume_optimizer_override",False):
                    raise ValueError(f"Exact resume mismatch: {name}; use --resume_optimizer_override only to change lr/grad_clip")
        if "optimizer" not in state or "rng" not in state or "scheduler" not in state:
            raise ValueError("Exact resume requires optimizer, scheduler and RNG state")
        optimizer.load_state_dict(state["optimizer"])
        if scheduler is None:
            raise ValueError("Exact resume requires a scheduler")
        scheduler.load_state_dict(state["scheduler"])
        if getattr(args,"resume_optimizer_override",False):
            reconfigure_learning_rate(optimizer,scheduler,args.lr)
            print(f"RESUME_OPTIMIZER_OVERRIDE old_base_lr={recorded.get('lr')} new_base_lr={args.lr} "
                  f"grad_clip={args.grad_clip} effective_lr={[g['lr'] for g in optimizer.param_groups]}; "
                  "Adam moments, completed epochs and scheduler phase retained",flush=True)
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
        prediction=model(registered_lr,msi)
        if not bool(torch.isfinite(prediction).all()):
            raise FloatingPointError("PMI-RFCoNet validation prediction is nonfinite; best/last checkpoint not updated")
        reg.update(calc_metrics(prediction, gt, args.scale_ratio))
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
                prediction=model(lr_deformed,msi)
                if not bool(torch.isfinite(prediction).all()):
                    raise FloatingPointError("PMI-RFCoNet warped validation prediction is nonfinite; checkpoint not updated")
                warp.update(calc_metrics(prediction, gt, args.scale_ratio))
    return reg.average(), warp.average()


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(row))
        if stream.tell() == 0:
            writer.writeheader()
        writer.writerow(row)


def main(stage="registered", argv=None):
    if stage not in ("registered", "mixed"):
        raise ValueError(f"Unsupported PMI-RFCoNet stage: {stage}")
    args = parser(stage, argv)
    args.stage = stage
    args.validation_interval = args.validation_interval or DATASET_VALIDATION_INTERVAL[args.dataset]
    if min(args.epochs, args.batch_size, args.validation_interval, args.validation_cases,
           args.mid_channels, args.num_fc) < 1:
        raise ValueError("Epochs, batch_size, validation interval/cases, mid_channels and num_fc must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.grad_clip < 0 or not 0 < args.lr_gamma <= 1:
        raise ValueError("Invalid optimizer/loss settings")
    if args.lr_milestones != sorted(set(args.lr_milestones)) or min(args.lr_milestones) < 1:
        raise ValueError("lr_milestones must be positive, unique and increasing")
    if (args.image_size,args.patch_size,args.stride,args.scale_ratio)!=(128,64,32,4):
        raise ValueError("Shared PMI-RFCoNet benchmark requires (eval,patch,stride,scale)=(128,64,32,4)")
    if not 0 <= args.registered_probability <= 1 or not 0 <= args.drop < 1:
        raise ValueError("Invalid identity probability or dropout")
    if args.max_train_batches < 0:
        raise ValueError("--max_train_batches must be >= 0")
    if args.degradation_mode != "physical":
        raise ValueError("PMI-RFCoNet unified stages require the shared physical degradation")
    if args.repair_bn_stats and not args.resume:
        raise ValueError("--repair_bn_stats requires --resume")
    if args.bn_calibration_batches<1:
        raise ValueError("bn_calibration_batches must be positive")
    if args.resume_optimizer_override and not args.resume:
        raise ValueError("--resume_optimizer_override requires --resume")
    if stage == "registered" and args.init_checkpoint:
        raise ValueError("Registered stage starts from scratch; use --resume to continue")

    set_seed(args.seed)
    device = resolve_device(args.device)
    train_set, val_set, _, info = build_datasets(build_shared_cfg(args), include_validation=True)
    if not len(train_set) or not len(val_set):
        raise ValueError("PMI-RFCoNet benchmark splits cannot be empty")
    loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                        num_workers=args.num_workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(val_set, batch_size=1, shuffle=False, num_workers=0)
    p0 = train_set.degradation_operator.to(device)
    fingerprint = sensor_fingerprint(info)
    model = build_model(info, args, device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, 0.999)
    )

    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=args.lr_milestones, gamma=args.lr_gamma
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
                                fingerprint=fingerprint, optimizer=optimizer, scheduler=scheduler)
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
        print(f"Initialized mixed PMI-RFCoNet from registered checkpoint: {init_path}", flush=True)

    if args.resume and state.get("_bn_recalibration_required"):
        repair=state["_bn_recalibration_required"]
        repair["source_checkpoint"]=str(args.resume)
        rng=snapshot_rng(geometry_gen)
        calibration_gen=make_generator(device,args.seed+88001)
        try:
            set_seed(args.seed+88001)
            calibration_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,num_workers=0,
                generator=torch.Generator().manual_seed(args.seed+88001))
            repair["calibration_batches"]=recalibrate_bn(
                model,calibration_loader,device,
                lambda gt:deformed_observation(gt,p0,args,calibration_gen,mixed=(stage=="mixed"))[0],
                args.bn_calibration_batches)
        finally:
            restore_rng(rng,geometry_gen)
        args.bn_stats_repair_info=repair
        # Old validation scores used corrupted population statistics; start a
        # new monitor, preserving the old best file before the first overwrite.
        best_psnr,bad_validations=-float("inf"),0
        old_best=ckpt_dir/"best.pth.tar"
        backup=ckpt_dir/"best_before_bn_repair.pth.tar"
        if old_best.is_file() and not backup.exists():
            shutil.copy2(old_best,backup)
        (ckpt_dir/"bn_repair_report.json").write_text(json.dumps(repair,indent=2),encoding="utf-8")
        torch.save(checkpoint_payload(model,optimizer,scheduler,args,start_epoch-1,best_psnr,
            bad_validations,fingerprint,geometry_gen),ckpt_dir/"repaired_resume.pth.tar")
        print(f"BN_STATS_RECALIBRATED {repair}; validation monitor reset; source checkpoint unchanged",flush=True)
    elif args.resume and state.get("bn_stats_repair"):
        args.bn_stats_repair_info=state["bn_stats_repair"]

    protocol = {
        "method": "PMI-RFCoNet", "architecture_id": ARCHITECTURE_ID,
        "upstream_blob_sha": UPSTREAM_BLOB_SHA,
        "upstream_commit": UPSTREAM_COMMIT,
        "dataset": args.dataset, "stage": stage,
        "hsi_channels": model.hsi_channels, "msi_channels": model.msi_channels,
        "sensor_fingerprint": fingerprint,
        "training_objective": "mean L1(prediction, HR-HSI), released supervised objective",
        "optimizer": "Adam", "scheduler": "MultiStepLR",
        "parameters": sum(p.numel() for p in model.parameters()), "args": vars(args),
    }
    (ckpt_dir / "run_protocol.txt").write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    print(json.dumps(protocol, indent=2), flush=True)

    for epoch in range(start_epoch, args.epochs):
        model.train()
        sum_total, count, identities = 0.0, 0, 0
        progress = tqdm(loader, desc=f"PMI-RFCoNet {stage} {epoch + 1}/{args.epochs}",
                        dynamic_ncols=True, mininterval=1.0, disable=None)
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
            if not all(bool(torch.isfinite(value).all()) for value in (lr_hsi,msi,gt)):
                fail_training(model,optimizer,lr_hsi,msi,gt,directory=ckpt_dir/"failure",args=args,
                    epoch=epoch+1,batch_idx=batch_idx,reason="nonfinite observation/target",replay=False)
            prediction = model(lr_hsi, msi)
            invalid_bn=nonfinite_bn_buffers(model)
            if invalid_bn:
                fail_training(model,optimizer,lr_hsi,msi,gt,directory=ckpt_dir/"failure",args=args,
                    epoch=epoch+1,batch_idx=batch_idx,
                    reason=f"nonfinite BatchNorm statistics: {invalid_bn}")
            loss = torch.nn.functional.l1_loss(prediction, gt)
            if not torch.isfinite(loss):
                fail_training(model,optimizer,lr_hsi,msi,gt,directory=ckpt_dir/"failure",args=args,
                    epoch=epoch+1,batch_idx=batch_idx,reason="nonfinite prediction/loss")
            loss.backward()
            try:
                grad_norm=torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clip if args.grad_clip > 0 else float("inf"),
                    error_if_nonfinite=True,
                )
            except RuntimeError as exc:
                if "non-finite" not in str(exc):
                    raise
                fail_training(model,optimizer,lr_hsi,msi,gt,directory=ckpt_dir/"failure",args=args,
                    epoch=epoch+1,batch_idx=batch_idx,reason="nonfinite gradient norm",replay=False)
            optimizer.step()
            b = int(gt.shape[0])
            count += b
            identities += identity_count
            sum_total += float(loss.detach()) * b
            progress.set_postfix(loss=f"{sum_total/max(count,1):.5f}",
                                 identity=f"{identities/max(count,1):.3f}", refresh=False)

        mean = {
            "train_total": sum_total / max(count, 1),
            "lr": optimizer.param_groups[0]["lr"],
            "identity_fraction": identities / max(count, 1),
        }
        progress.close()
        scheduler.step()
        print(f"PMI-RFCoNet {stage} epoch={epoch+1} {mean}", flush=True)
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
                    model, optimizer, scheduler, args, epoch, best_psnr, bad_validations,
                    fingerprint, geometry_gen
                ), ckpt_dir / "best.pth.tar")
            if args.early_stop_patience > 0 and bad_validations >= args.early_stop_patience:
                print("PMI-RFCoNet validation early stop", flush=True)
                torch.save(checkpoint_payload(
                    model, optimizer, scheduler, args, epoch, best_psnr, bad_validations,
                    fingerprint, geometry_gen
                ), ckpt_dir / "last.pth.tar")
                break

        torch.save(checkpoint_payload(
            model, optimizer, scheduler, args, epoch, best_psnr, bad_validations,
            fingerprint, geometry_gen
        ), ckpt_dir / "last.pth.tar")


if __name__ == "__main__":
    main()
