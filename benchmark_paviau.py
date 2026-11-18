"""Benchmark local PaviaU checkpoints for the paper; see BENCHMARK_PAVIAU.md."""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import hashlib
import importlib
import json
import os
from pathlib import Path
import random
from statistics import mean, median, stdev
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parent
METHODS = ("CNMF", "HySure", "MoG-DCN", "PSRT", "ESFS", "CLSNet", "EMR-Diff", "M2DTN",
           "PMI-RFCoNet", "MoE-PNP", "SA-RFM", "UAFL", "ours")
CLASSICAL = frozenset(("CNMF", "HySure"))
REGISTERED = CLASSICAL | {"MoG-DCN", "PSRT", "ESFS", "CLSNet", "EMR-Diff"}
REGISTERED_ONLY = CLASSICAL | {"MoG-DCN", "ESFS"}
LR_SHAPE, MSI_SHAPE, OUT_SHAPE = (1, 103, 32, 32), (1, 4, 128, 128), (1, 103, 128, 128)
FIELDS = (
    "method", "dataset", "status", "weight_source", "checkpoint_stage", "parameters", "params_M",
    "flops", "FLOPs_G", "flops_status", "latency_ms_median", "latency_ms_mean",
    "latency_ms_std", "device", "hardware", "cpu_threads", "precision", "warmup", "repeats", "seed",
    "input_sha256", "diffusion_steps", "geometry_steps", "ode_steps", "backbone_width",
    "checkpoint", "flops_uncounted_ops", "error",
)


def runtime(device, threads, seed):
    import numpy as np
    import torch

    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("Use a CPU or CUDA device")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu explicitly for a CPU run")
    torch.set_num_threads(threads)
    torch.set_float32_matmul_precision("highest")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = False
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    return device


def verify_shape(tensor, expected, label):
    if tuple(tensor.shape) != expected:
        raise ValueError(f"{label} shape {tuple(tensor.shape)} differs from PaviaU benchmark {expected}")


def parameter_count(modules):
    params = {id(p): p for module in modules for p in module.parameters()}
    return sum(p.numel() for p in params.values())


def dense_flops(forward):
    """Count multiply-add work of dispatched dense ops; 1 MAC = 2 FLOPs."""
    import torch
    from torch.utils.flop_counter import FlopCounterMode, sdpa_flop_count

    custom = {}
    cpu_sdp = getattr(torch.ops.aten, "_scaled_dot_product_flash_attention_for_cpu", None)
    if cpu_sdp is not None:
        def sdp(q, k, v, *args, out_shape=None, **kwargs):
            return sdpa_flop_count(q, k, v)
        custom[cpu_sdp] = sdp

    deform = getattr(torch.ops.torchvision, "deform_conv2d", None)
    if deform is not None:
        def deform_dense(x, weight, *args, out_shape=None, **kwargs):
            batch, out_channels, h, w = out_shape
            return 2 * batch * out_channels * h * w * weight[1] * weight[2] * weight[3]
        custom[deform] = deform_dense

    class CounterWithCoverage(FlopCounterMode):
        def __init__(self):
            super().__init__(display=False, custom_mapping=custom)
            self.uncounted = Counter()

        def _count_flops(self, op, out, args, kwargs):
            name = str(op)
            if op not in self.flop_registry and (
                not name.startswith("aten.") or any(key in name for key in (
                    "fft", "grid_sampler", "attention", "linalg", "upsample", "sort"
                ))
            ):
                self.uncounted[name] += 1
            elif deform is not None and op == deform:
                self.uncounted["torchvision.deform_conv2d_sampling"] += 1
            return super()._count_flops(op, out, args, kwargs)

    # Expose the attention matmuls when the fused MHA inference path would
    # make them invisible; do this only for counting, never during timing.
    mha_fastpath = torch.backends.mha.get_fastpath_enabled()
    try:
        torch.backends.mha.set_fastpath_enabled(False)
        with CounterWithCoverage() as counter:
            forward()
    finally:
        torch.backends.mha.set_fastpath_enabled(mha_fastpath)
    counts = counter.get_total_flops()
    if counts <= 0:
        raise RuntimeError("No supported FLOPs counted for this neural model")
    return counts, "; ".join(f"{name}:{n}" for name, n in sorted(counter.uncounted.items()))


def measure(forward, modules, device, *, warmup, repeats, threads, seed, classical=False):
    import platform
    import numpy as np
    import torch

    def synchronize():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    with torch.no_grad():
        torch.manual_seed(seed)
        for _ in range(warmup):
            forward()
        synchronize()
        times = []
        for _ in range(repeats):
            start = time.perf_counter()
            result = forward()
            synchronize()
            times.append((time.perf_counter() - start) * 1000)
            # Do not retain a previous output during the next timed inference.
            if len(times) < repeats:
                del result
        verify_shape(result, OUT_SHAPE, "prediction")
        finite = np.isfinite(result).all() if classical else bool(torch.isfinite(result).all())
        if not finite:
            raise ValueError("Nonfinite PaviaU prediction")
        del result
        hardware = torch.cuda.get_device_name(device) if device.type == "cuda" else (
            platform.processor() or platform.machine()
        )
        params = parameter_count(modules)
        row = {
            "device": str(device), "hardware": hardware, "cpu_threads": threads,
            "precision": "native solver" if classical else "FP32 (TF32 off)",
            "warmup": warmup, "repeats": repeats, "parameters": params, "params_M": params / 1e6,
            "latency_ms_median": median(times), "latency_ms_mean": mean(times),
            "latency_ms_std": stdev(times) if len(times) > 1 else None,
            "flops_status": "not_applicable" if classical else "estimated_dense_ops",
        }
        if not classical:
            try:
                torch.manual_seed(seed)
                flops, uncounted = dense_flops(forward)
                row.update(flops=flops, FLOPs_G=flops / 1e9, flops_uncounted_ops=uncounted)
            except Exception as exc:
                row.update(flops_status="unavailable", error=f"FLOPs: {type(exc).__name__}: {exc}")
        return row


def load_pair(path, device):
    import numpy as np
    import torch

    with np.load(path, allow_pickle=False) as data:
        lr, msi, srf = (torch.from_numpy(data[key]) for key in ("lr_hsi", "hr_msi", "srf"))
    for value, shape, label in ((lr, LR_SHAPE, "LR-HSI"), (msi, MSI_SHAPE, "HR-MSI"),
                                (srf, (4, 103), "SRF")):
        verify_shape(value, shape, label)
        if value.dtype != torch.float32 or not bool(torch.isfinite(value).all()):
            raise ValueError(f"Invalid saved PaviaU {label}")
    return lr.to(device), msi.to(device), srf


def test_args(args):
    return SimpleNamespace(
        dataset="PaviaU", data_root=str(args.data_root), device=args.device, seed=args.seed,
        eval_seed=args.seed, image_size=128, patch_size=64, stride=32, scale_ratio=4,
        degradation_mode="physical", mtf_nyquist=0.2, psf_truncate=3.0,
        degradation_sigma=2.0, degradation_kernel_size=5, msi_mode="srf", srf_path="",
        wavelength_root="./data/wavelengths", wavelength_path="", srf_interp="pchip",
        srf_band_set="auto", max_translation=4.0, max_rotation_deg=2.0, max_local_px=4.0,
        control_grid=5, min_jacobian=0.5, registered_probability=0.10,
    )


def build_test_set(cfg):
    from comparison.UAFL.common import build_shared_cfg
    from data_loader import build_datasets

    _, _, test_set, info = build_datasets(build_shared_cfg(cfg), include_validation=True)
    if len(test_set) != 1:
        raise ValueError("Expected one held-out PaviaU center crop")
    return test_set, info


def save_shared_pair(args, path):
    import numpy as np
    from comparison.UAFL.common import require_srf_weights

    runtime("cpu", args.cpu_threads, args.seed)
    # The existing SRF registry uses paths relative to the comparison root.
    cwd = Path.cwd()
    try:
        os.chdir(ROOT)
        test_set, info = build_test_set(test_args(args))
    finally:
        os.chdir(cwd)
    sample = test_set[0]
    arrays = {
        "lr_hsi": sample["lr_hsi"].unsqueeze(0).float().numpy(),
        "hr_msi": sample["hr_msi"].unsqueeze(0).float().numpy(),
        "srf": require_srf_weights(info).float().contiguous().numpy(),
    }
    np.savez(path, **arrays)
    load_pair(path, "cpu")  # Validate before launching any worker.
    sha = hashlib.sha256()
    for array in arrays.values():
        sha.update(array.tobytes())
    return sha.hexdigest()


def initialized_baseline(method, cfg, srf, device):
    """Build the same standard PaviaU modules as the test entry points."""
    info = {"n_bands": srf.shape[1], "n_select_bands": srf.shape[0], "srf_weights": srf,
            "scale_ratio": cfg.scale_ratio}
    if method in ("UAFL", "EMR-Diff"):
        from test_fixed_deformation import _load_evaluator

        evaluator = _load_evaluator(method)
        if method == "EMR-Diff":
            trainer = evaluator.build_trainer(cfg)
            trainer.Net.eval()
            return trainer._reconstruct, [trainer.Net], {
                "backbone_width": trainer.backbone_width, "diffusion_steps": trainer.num_timesteps,
            }
        model = evaluator.build_uafl(info["n_select_bands"]).to(device).eval()

        def predict(lr, msi):
            return model(evaluator.upsample_lr_hsi(lr, msi.shape[-2:]), msi)

        return predict, [model], {}

    if method in ("CLSNet", "M2DTN", "MoE-PNP"):
        model = importlib.import_module(f"comparison.{method}.common").model_for_info(info, cfg, device)
    else:
        build = importlib.import_module(f"comparison.{method}.model").build_model
        model = build(info).to(device) if method in ("MoG-DCN", "ESFS") else build(info, cfg, device)
    details = {"ode_steps": model.ode_steps} if method == "SA-RFM" else {}
    return model.eval(), [model], details


def benchmark_baseline(args):
    import numpy as np
    import torch

    method = args.methods[0]
    classical = method in CLASSICAL
    device = runtime("cpu" if classical else args.device, args.cpu_threads, args.seed)
    lr, msi, srf = load_pair(args._pair, device)
    options = dict(warmup=args.classical_warmup if classical else args.warmup,
                   repeats=args.classical_repeats if classical else args.repeats,
                   threads=args.cpu_threads, seed=args.seed)
    if classical:
        from threadpoolctl import threadpool_limits
        from comparison.traditional.operators import PhysicalOperator
        from comparison.traditional.options import DEFAULTS as opts

        module = importlib.import_module(f"comparison.{method}.model")
        if method == "CNMF":
            config = module.CNMFConfig(opts["cnmf_iterations"], opts["cnmf_endmembers"], args.seed)
        else:
            config = module.HySureConfig(
                iterations=opts["hysure_iterations"], rank=opts["hysure_rank"],
                lambda_phi=opts["hysure_lambda_phi"], lambda_m=opts["hysure_lambda_m"],
                cg_maxiter=opts["hysure_cg_maxiter"], cg_rtol=opts["hysure_cg_rtol"],
            )
        operator = PhysicalOperator(4, 0.2, 3.0)
        hsi_np, msi_np, srf_np = lr[0].numpy(), msi[0].numpy(), srf.double().numpy()

        def forward():
            return np.asarray(module.fuse(hsi_np, msi_np, srf_np, operator, config)[0])[None]

        with threadpool_limits(limits=args.cpu_threads):
            return measure(forward, [], device, classical=True, **options)

    cfg = test_args(args)
    cfg.device = str(device)
    cfg.mode = args._stage
    cfg.checkpoint = str(args._checkpoint) if args._checkpoint else ""
    cfg.hidden_channels = args.emr_width if method == "EMR-Diff" else None
    modules = []
    if args.no_checkpoints:
        predict, modules, details = initialized_baseline(method, cfg, srf, device)
    elif method in ("MoG-DCN", "ESFS"):
        from comparison.registered_fusion.runner import make_model, load_state

        _, info = build_test_set(cfg)
        if not torch.equal(torch.as_tensor(info["srf_weights"]), srf):
            raise ValueError("PaviaU model/dataset SRF differs from the benchmark pair")
        predict, _, metadata = make_model(method, cfg, info)
        load_state(cfg.checkpoint, predict, metadata)
        predict.eval()
        details = {}
    else:
        from test_fixed_deformation import MODEL_BUILDERS

        builder = MODEL_BUILDERS[method]
        data, predict, _, built_device, details = builder(cfg, profile_modules=modules) if method in (
            "UAFL", "EMR-Diff"
        ) else builder(cfg)
        if torch.device(built_device) != device:
            raise RuntimeError(f"{method} resolved a different device: {built_device}")
        if getattr(data, "srf_weights", None) is not None and not torch.equal(
            torch.as_tensor(data.srf_weights).cpu(), srf
        ):
            raise ValueError("PaviaU model/dataset SRF differs from the benchmark pair")
    if isinstance(predict, torch.nn.Module) and not modules:
        modules.append(predict)
    if not modules:
        raise RuntimeError(f"No parameter modules exposed for {method}")
    runtime(device, args.cpu_threads, args.seed)
    row = measure(lambda: predict(lr, msi), modules, device, **options)
    for key in ("diffusion_steps", "ode_steps", "backbone_width"):
        if key in details:
            row[key] = details[key]
    return row


def parse_args(argv=None):
    from emr_configuration import validate_hidden_channels

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--s2diff_root", type=Path, default=ROOT.parent / "S2Diff-MH")
    p.add_argument("--data_root", type=Path, help="Folder containing PaviaU.mat; default: detect in either repo")
    p.add_argument("--output_dir", type=Path)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--repeats", type=int, default=20)
    p.add_argument("--classical_warmup", type=int, default=1)
    p.add_argument("--classical_repeats", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--emr_width", type=validate_hidden_channels)
    p.add_argument("--checkpoint", action="append", default=[], metavar="METHOD=PATH")
    p.add_argument("--model_stage", action="append", default=[], metavar="METHOD=registered|mixed")
    p.add_argument("--no_checkpoints", action="store_true",
                   help="Benchmark initialized models with the standard PaviaU architecture")
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--_pair", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_result", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_checkpoint", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--_stage", help=argparse.SUPPRESS)
    p.add_argument("--ours_args", nargs=argparse.REMAINDER, default=[], help="S2Diff-MH pipeline arguments; put last")
    args = p.parse_args(argv)
    if min(args.repeats, args.classical_repeats, args.cpu_threads) < 1 or min(
        args.seed, args.warmup, args.classical_warmup
    ) < 0:
        p.error("repeats and threads must be positive; seed and warmups must be nonnegative")
    if args.no_checkpoints and (args.checkpoint or args.model_stage or args._checkpoint):
        p.error("--no_checkpoints cannot be combined with checkpoint or model stage overrides")
    return args


def overrides(entries, *, field):
    values = {}
    for entry in entries:
        name, sep, value = entry.partition("=")
        if not sep or name not in set(METHODS) - CLASSICAL - {"ours"} or not value or name in values:
            raise ValueError(f"Invalid {field} override: {entry!r}")
        if field == "model_stage" and (value not in ("registered", "mixed") or
                                       name in REGISTERED_ONLY and value != "registered"):
            raise ValueError(f"Unsupported checkpoint stage: {entry!r}")
        values[name] = value
    return values


def resolve_checkpoint(method, stage, explicit, emr_width):
    from experiment_protocol import checkpoint_path, resolve_model

    if explicit:
        return Path(explicit).expanduser().resolve()
    width = emr_width if method == "EMR-Diff" else None
    path = checkpoint_path(resolve_model(method), "PaviaU", stage, width)
    if method == "EMR-Diff" and width is None and not path.is_file():
        candidates = sorted(p for p in path.parent.glob("best_base*.pth.tar") if p.is_file())
        if len(candidates) == 1:
            return candidates[0]
        if len(candidates) > 1:
            raise ValueError("Multiple alternative EMR-Diff widths; select one with --emr_width")
    return path


def write_summary(output, rows):
    path = output / "summary.csv"
    pending = path.with_suffix(".csv.tmp")
    with pending.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    pending.replace(path)
    return path


def main(argv=None):
    args = parse_args(argv)
    if args._pair:
        row = benchmark_baseline(args)
        args._result.write_text(json.dumps(row, allow_nan=False), encoding="utf-8")
        return 0

    args.s2diff_root = args.s2diff_root.expanduser().resolve()
    if args.data_root is None:
        dirs = [ROOT / "data/raw", args.s2diff_root / "data/raw"]
        args.data_root = next((d for d in dirs if (d / "PaviaU.mat").is_file()), dirs[0])
    args.data_root = args.data_root.expanduser().resolve()
    if args.output_dir is None:
        name = "benchmark_paviau_initialized" if args.no_checkpoints else "benchmark_paviau"
        args.output_dir = ROOT / "outputs" / name
    args.output_dir = args.output_dir.expanduser().resolve()
    checkpoints = overrides(args.checkpoint, field="checkpoint")
    stages = overrides(args.model_stage, field="model_stage")
    methods = tuple(dict.fromkeys(args.methods))
    print(f"PAVIAU_BENCHMARK data={args.data_root} S2Diff-MH={args.s2diff_root} device={args.device}", flush=True)
    if args.dry_run:
        for method in methods:
            stage = stages.get(method, "registered" if method in REGISTERED else "mixed")
            if method in CLASSICAL:
                source = "CPU solver"
            elif args.no_checkpoints:
                source = "initialized (no checkpoint)"
                if method == "EMR-Diff":
                    source += f", backbone_width={args.emr_width or 64}"
                stage = "-"
            elif method == "ours":
                source = args.s2diff_root / Path(__file__).name
            else:
                try:
                    source = resolve_checkpoint(method, stage, checkpoints.get(method), args.emr_width)
                except ValueError as exc:
                    source = str(exc)
            print(f"PLAN {method:<13} {stage:<10} {source}", flush=True)
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logs = args.output_dir / "logs"
    logs.mkdir(exist_ok=True)
    rows = []
    with TemporaryDirectory(prefix="benchmark-paviau-") as directory:
        pair = Path(directory) / "pair.npz"
        pair_sha = save_shared_pair(args, pair)
        for method in methods:
            stage = stages.get(method, "registered" if method in REGISTERED else "mixed")
            weight_source = ("not_applicable" if method in CLASSICAL else
                             "initialized" if args.no_checkpoints else "checkpoint")
            row = dict(method=method, dataset="PaviaU", weight_source=weight_source,
                       checkpoint_stage=stage if weight_source == "checkpoint" else "",
                       input_sha256=pair_sha, seed=args.seed)
            result = Path(directory) / f"{method}.json"
            workdir = args.s2diff_root if method == "ours" else ROOT
            worker = workdir / Path(__file__).name
            command = [sys.executable, str(worker), "--_pair", str(pair), "--_result", str(result),
                       "--device", args.device, "--cpu_threads", str(args.cpu_threads),
                       "--warmup", str(args.warmup), "--repeats", str(args.repeats),
                       "--seed", str(args.seed)]
            if args.no_checkpoints:
                command.append("--no_checkpoints")
            log = logs / f"{method}.log"
            try:
                if method == "ours":
                    if not worker.is_file():
                        raise FileNotFoundError(f"Install the S2Diff-MH benchmark script: {worker}")
                    command += ["--_runner", str(Path(__file__).resolve()), "--ours_args", *args.ours_args]
                else:
                    command += ["--methods", method, "--data_root", str(args.data_root),
                                "--_stage", stage, "--classical_warmup", str(args.classical_warmup),
                                "--classical_repeats", str(args.classical_repeats)]
                    if weight_source == "checkpoint":
                        path = resolve_checkpoint(method, stage, checkpoints.get(method), args.emr_width)
                        row["checkpoint"] = str(path)
                        if not path.is_file():
                            raise FileNotFoundError(f"Checkpoint not found: {path}")
                        command += ["--_checkpoint", str(path)]
                    if method == "EMR-Diff" and args.emr_width is not None:
                        command += ["--emr_width", str(args.emr_width)]
                print(f"START {method} ({stage if weight_source == 'checkpoint' else weight_source})", flush=True)
                with log.open("w", encoding="utf-8") as handle:
                    proc = subprocess.run(command, cwd=workdir, stdout=handle, stderr=subprocess.STDOUT)
                if proc.returncode:
                    lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    cause = lines[-1] if lines else f"exit status {proc.returncode}"
                    raise RuntimeError(f"{cause} (see {log})")
                row.update(json.loads(result.read_text(encoding="utf-8")))
                row["status"] = "partial" if row.get("error") else "ok"
            except (OSError, ValueError, RuntimeError) as exc:
                row.update(status="failed", error=str(exc))
            if row["status"] != "ok":
                print(f"{row['status'].upper()} {method}: {row.get('error', '')}", flush=True)
            rows.append(row)
            write_summary(args.output_dir, rows)

    print(f"{'Method':<14} {'Params (M)':>12} {'FLOPs (G)':>12} {'Median (ms)':>14}  Device / status")
    for row in rows:
        values = ["—" if row.get(k) is None else f"{row[k]:.4f}"
                  for k in ("params_M", "FLOPs_G", "latency_ms_median")]
        print(f"{row['method']:<14} {values[0]:>12} {values[1]:>12} {values[2]:>14}"
              f"  {row.get('device', '')} / {row['status']}")
    print(f"SUMMARY_CSV={args.output_dir / 'summary.csv'}", flush=True)
    return int(any(row["status"] != "ok" for row in rows))


if __name__ == "__main__":
    raise SystemExit(main())
