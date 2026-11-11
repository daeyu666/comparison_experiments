"""Read-only audit of one MoG-DCN checkpoint on the registered val/test splits."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison.registered_fusion.runner import (
    batch_to_device, build_datasets, build_shared_cfg, load_state, make_model,
    parser, run_dir, set_seed,
)
from experiment_protocol import resolve_model
from metrics import calc_metrics

METHOD = "MoG-DCN"


def rms(x):
    return float(x.double().square().mean().sqrt())


def error_profile(pred, gt):
    """Break down the same clipped error used by the shared PSNR."""
    p, g = pred.double().clamp(0, 1), gt.double().clamp(0, 1)
    error = (p - g).square()
    pixel_error = error.sum(dim=1)
    sse = float(pixel_error.sum())
    amplitude = (p.norm(dim=1) - g.norm(dim=1)).square()
    # ||p-g||² = (||p||-||g||)² + 2||p||||g||(1-cos theta).
    fraction = min(float(amplitude.sum()) / sse, 1.0) if sse else None
    flat = pixel_error.flatten()
    count = max(1, math.ceil(0.01 * flat.numel()))
    band_mse = error.mean(dim=(0, 2, 3))
    values, indices = band_mse.topk(min(5, g.shape[1]))
    return {
        "raw_rmse": rms(pred.double() - gt.double()),
        "pred_below_zero_fraction": float((pred < 0).float().mean()),
        "pred_above_one_fraction": float((pred > 1).float().mean()),
        "gt_rms": rms(g),
        "relative_rmse": float(error.mean().sqrt()) / max(rms(g), 1e-12),
        "spectral_magnitude_fraction_of_sse": fraction,
        "top_1pct_pixels_fraction_of_sse": float(flat.topk(count).values.sum()) / sse if sse else None,
        "worst_bands": [{"band_1based": int(i) + 1, "rmse": math.sqrt(float(v))}
                        for v, i in zip(values, indices)],
    }


def stats(x):
    return {"rms": rms(x), "max_abs": float(x.abs().amax())}


def trace_hooks(model, lr, msi):
    """Read active operators without changing the model or prediction."""
    trace = {k: [] for k in ("spatial_forward", "spatial_back", "spectral_forward", "spectral_back")}
    observations = {"spatial_forward": lr, "spectral_forward": msi}
    handles = []

    def callback(name):
        def record(_module, inputs, output):
            row = {"input": stats(inputs[0]), "output": stats(output)}
            if name in observations:
                row["observation_residual_rms"] = rms(output - observations[name])
            trace[name].append(row)
        return record

    for name in trace:
        handles.append(getattr(model, name).register_forward_hook(callback(name)))
    return trace, handles


def predict_in_training_tiles(model, lr, msi, tile, scale):
    h, w = msi.shape[-2:]
    if tile % scale or h % tile or w % tile:
        raise ValueError("The evaluation input must split into LR-aligned training tiles")
    return torch.cat([
        torch.cat([
            model(lr[..., y // scale:(y + tile) // scale, x // scale:(x + tile) // scale],
                  msi[..., y:y + tile, x:x + tile])
            for x in range(0, w, tile)
        ], dim=-1)
        for y in range(0, h, tile)
    ], dim=-2)


@torch.inference_mode()
def audit_split(model, split, device, args):
    # Houston13/PaviaU each have exactly one patch per held-out split.
    if len(split) != 1:
        raise ValueError("Expected one registered patch in this diagnostic split")
    lr, msi, gt = batch_to_device(next(iter(DataLoader(split, batch_size=1))), device)
    trace, handles = trace_hooks(model, lr, msi)
    try:
        pred = model(lr, msi)
    finally:
        for handle in handles:
            handle.remove()
    if not bool(torch.isfinite(pred).all()):
        raise FloatingPointError("Nonfinite MoG-DCN prediction on the selected split")
    baseline = F.interpolate(lr, size=gt.shape[-2:], mode="bicubic", align_corners=False)
    result = {
        "official": calc_metrics(pred, gt, args.scale_ratio),
        "bicubic": calc_metrics(baseline, gt, args.scale_ratio),
        "error": error_profile(pred, gt),
        "operator_trace": trace,
    }
    if args.check_tiled:
        tiled = predict_in_training_tiles(model, lr, msi, args.patch_size, args.scale_ratio)
        if not bool(torch.isfinite(tiled).all()):
            raise FloatingPointError("Nonfinite MoG-DCN prediction in the tile probe")
        result["training_size_tile_probe"] = {
            "metrics": calc_metrics(tiled, gt, args.scale_ratio),
            "rmse_vs_whole_prediction": rms(tiled - pred),
        }
    return result


def read_epoch_log(path, epoch):
    if not path.is_file():
        return None
    with path.open(newline="", encoding="utf-8") as stream:
        rows = [row for row in csv.DictReader(stream) if int(row["epoch"]) == epoch]
    if len(rows) > 1:
        raise ValueError(f"Multiple validation records for epoch {epoch} in {path}; pass the matching log")
    if not rows:
        return None
    return {"PSNR": float(rows[0]["reg_PSNR"]), "SAM": float(rows[0]["reg_SAM"])}


def main(argv=None):
    p = parser(METHOD)
    p.set_defaults(dataset="Houston13")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--validation_log", type=Path)
    p.add_argument("--check_tiled", action="store_true",
                   help="Diagnostic only: reuse the observed pair, but infer in training-size tiles")
    p.add_argument("--output_json", type=Path)
    args = p.parse_args(argv)
    if args.dataset not in ("Houston13", "PaviaU", "Botswana"):
        p.error("Choose a single-patch validation/test dataset: Houston13, PaviaU or Botswana")
    set_seed(1234)
    adapter = resolve_model(METHOD)
    checkpoint = args.checkpoint or run_dir(adapter, "checkpoints", args) / "best.pth.tar"
    if not checkpoint.is_file():
        p.error(f"Checkpoint not found: {checkpoint}")
    cfg = build_shared_cfg(args)
    _, val, test, info = build_datasets(cfg, include_validation=True)
    model, device, metadata = make_model(METHOD, args, info)
    state = load_state(checkpoint, model, metadata)
    model.eval()
    epoch = int(state["epoch"])
    path = args.validation_log or run_dir(adapter, "logs", args) / "validation_history.csv"
    logged = read_epoch_log(path, epoch)
    result = {
        **metadata, "checkpoint": str(checkpoint.resolve()), "checkpoint_epoch": epoch,
        "recorded_best_psnr": float(state["best_psnr"]) if math.isfinite(state["best_psnr"]) else None,
        "data_file": str((Path(args.data_root) / cfg.datasets[args.dataset].file_name).resolve()),
        "validation_log": str(path.resolve()) if path.is_file() else None,
        "logged_validation_at_checkpoint_epoch": logged,
        "coefficients": {name: float(getattr(model, name).detach()) for name in
                         ("prior_step", "prior_penalty", "data_step", "data_penalty")},
    }
    del state
    for name, split in (("validation", val), ("test", test)):
        result[name] = {"rect": info[f"{name}_rect"], **audit_split(model, split, device, args)}
        summary = result[name]
        official, baseline = summary["official"], summary["bicubic"]
        print(f"SPLIT {name} PSNR={official['PSNR']:.6f} SAM={official['SAM']:.6f} "
              f"RMSE={official['RMSE']:.7f} BICUBIC_PSNR={baseline['PSNR']:.6f}", flush=True)
        print("ERROR " + json.dumps(summary["error"], allow_nan=False), flush=True)
        for operator, records in summary["operator_trace"].items():
            print(f"OPERATOR {name} {operator} " + json.dumps(records, allow_nan=False), flush=True)
        if "training_size_tile_probe" in summary:
            print(f"TILE_PROBE {name} " + json.dumps(summary["training_size_tile_probe"]), flush=True)
    if logged:
        measured = result["validation"]["official"]
        result["validation_minus_log"] = {key: measured[key] - logged[key] for key in logged}
    print(f"CHECKPOINT epoch={epoch} path={checkpoint.resolve()} best_psnr={result['recorded_best_psnr']} "
          f"recomputed_minus_logged={result.get('validation_minus_log')}", flush=True)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        print(f"MOG_SPLIT_DIAGNOSTIC={args.output_json}", flush=True)
    return result


if __name__ == "__main__":
    main()
