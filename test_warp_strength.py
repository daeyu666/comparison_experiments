"""Evaluate all mixed-trained methods on five paired random warp strengths."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DATASETS = ("PaviaU", "Houston13", "Chikusei", "CAVE", "Botswana", "Augsburg")
METHODS = ("M2DTN", "PMI-RFCoNet", "MoE-PNP", "SA-RFM", "UAFL", "ours")
STRENGTHS = (0.0, 0.25, 0.5, 0.75, 1.0)
METRICS = ("PSNR", "SAM", "RMSE", "ERGAS", "SSIM", "CC")
PROTOCOL = "paired-random-warp-strength-v1"
FIELDS = ("dataset", "method", "strength", "translation_max_hr_px", "rotation_max_deg",
          "local_max_hr_px", "status", "test_patches", "cases_per_patch", "evaluations",
          *METRICS, *(m + "_std" for m in METRICS), "seed", "eval_seed", "input_sha256",
          "weights", "srf_audit", "protocol", "error")


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    pending = path.with_suffix(".tmp.json")
    pending.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    pending.replace(path)


def update_array_hash(digest, name, array):
    import numpy as np
    array = np.ascontiguousarray(array, dtype=np.float32)
    digest.update(name.encode())
    digest.update(str(array.shape).encode())
    digest.update(array.tobytes())


def paired_geometries(height, width, sample, case, seed):
    """Sample at full strength, then scale the SAME admissible geometry."""
    import torch
    from comparison.UAFL.hsi_deformation import (
        SyntheticGeometry, jacobian_determinant, sample_synthetic_geometry,
    )
    local_seed = (seed + 70000 + sample * 1000003 + case * 9176) % (2**63 - 1)
    generator = torch.Generator(device="cpu").manual_seed(local_seed)
    for _ in range(64):
        base = sample_synthetic_geometry(
            height, width, device=torch.device("cpu"), dtype=torch.float32, generator=generator,
            max_translation=4.0, max_rotation_deg=2.0, max_local_px=4.0,
            control_grid=5, min_jacobian=0.5, local_strength_min_fraction=0.0,
            local_strength_max_fraction=1.0,
        )
        geometries = [SyntheticGeometry(**{k: getattr(base, k) * alpha for k in (
            "dx", "dy", "theta_deg", "control", "local_field")}) for alpha in STRENGTHS]
        if all(float(jacobian_determinant(g.local_field).min()) >= 0.5 for g in geometries):
            return geometries
    raise RuntimeError("Cannot draw a geometry admissible at every strength")


def prepare_inputs(job, directory):
    """CPU synthesis ONCE per dataset; every method reads identical observations."""
    import numpy as np
    import torch
    from comparison.UAFL.hsi_deformation import forward_warp
    from export_paper_predictions import dataset_and_info, model_args

    test, info = dataset_and_info(model_args(job["dataset"], job["data_root"], "cpu", job["seed"]))
    if not len(test):
        raise ValueError("Empty official test split")
    srf = np.asarray(info["srf_weights"], dtype=np.float32)
    digest = hashlib.sha256()
    protocol = dict(id=PROTOCOL, dataset=job["dataset"], seed=job["seed"], cases=job["cases"],
                    strengths=STRENGTHS, degradation="physical", scale=4, mtf_nyquist=0.2,
                    psf_truncate=3.0, control_grid=5, min_jacobian=0.5,
                    geometry="paired random dx/dy/theta and B-spline; base scaled by alpha",
                    aggregation="mean across patches per case, then mean/std across cases",
                    metrics="fixed_deformation_protocol.calc_metrics (S2 final-test formulas)")
    digest.update(json.dumps(protocol, sort_keys=True).encode())
    update_array_hash(digest, "srf", srf)
    paths = []
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(len(test)):
        item = test[index]
        gt, msi = item["gt"].float(), item["hr_msi"].float()
        lr0 = item["lr_hsi"].float()
        observations = torch.empty(len(STRENGTHS), job["cases"], *lr0.shape)
        observations[0] = lr0
        geometry_records = []
        for case in range(job["cases"]):
            geometries = paired_geometries(*gt.shape[-2:], index, case, job["seed"])
            for level, geometry in enumerate(geometries):
                if level:
                    with torch.no_grad():
                        observations[level, case] = test.degradation_operator.degrade(
                            forward_warp(gt[None], geometry))[0]
            base = geometries[-1]
            geometry_records.append(dict(case=case, dx=float(base.dx), dy=float(base.dy),
                                         rotation_deg=float(base.theta_deg),
                                         local_max_hr_px=float(torch.linalg.vector_norm(base.local_field, dim=1).max()),
                                         control=base.control.tolist()))
        arrays = dict(gt=gt.numpy(), hr_msi=msi.numpy(), observations=observations.numpy())
        for name, value in arrays.items():
            update_array_hash(digest, f"{index}/{name}", value)
        path = directory / f"patch_{index:05}.npz"
        np.savez(path, **arrays)
        paths.append(str(path))
        write_json(directory / f"geometry_{index:05}.json", geometry_records)
        if (index + 1) % 16 == 0 or index + 1 == len(test):
            print(f"SHARED_INPUTS {job['dataset']} {index + 1}/{len(test)}", flush=True)
    manifest = dict(protocol=protocol, srf=srf.tolist(), paths=paths, patches=len(test),
                    input_sha256=digest.hexdigest())
    write_json(directory / "manifest.json", manifest)
    return manifest


def worker(job):
    import numpy as np
    import torch
    from export_paper_predictions import (
        SRF_CHECKED, checkpoint_srf_audit, model_args, ours_paths, ours_predictor, sha256,
    )

    torch.set_num_threads(job["cpu_threads"])
    torch.manual_seed(job["seed"])
    manifest = json.loads(Path(job["manifest"]).read_text(encoding="utf-8"))
    srf = np.asarray(manifest["srf"], dtype=np.float32)
    device = torch.device(job["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; use --device cpu")
    # Use the same metric implementation in both repository environments.
    metric_fn = load_module("warp_strength_shared_metrics", ROOT / "fixed_deformation_protocol.py").calc_metrics
    if job["method"] == "ours":
        paths, _, _ = ours_paths(job)
    else:
        paths = {"model": Path(job["checkpoint"])}
    weights = {role: {"path": str(path), "sha256": sha256(path)} for role, path in paths.items()}
    identity = dict(protocol=PROTOCOL, input_sha256=manifest["input_sha256"], weights=weights,
                    method=job["method"], dataset=job["dataset"], eval_seed=job["eval_seed"],
                    ours_config=job.get("ours_config") if job["method"] == "ours" else None,
                    device=str(device), torch_version=str(torch.__version__))
    completed, pending = [], []
    for level, alpha in enumerate(STRENGTHS):
        path = Path(job["result_dir"]) / f"strength_{alpha:.2f}.json"
        if job["resume"] and path.is_file():
            saved = json.loads(path.read_text(encoding="utf-8"))
            if saved["identity"] != identity:
                raise ValueError(f"Different inputs/checkpoints/settings at {path}; use a new --output_dir")
            completed.append(saved["row"])
        else:
            pending.append((level, alpha, path))
    write_json(Path(job["worker_rows"]), completed)
    if not pending:
        print("All five strengths already completed", flush=True)
        return
    audit_status = "not_recorded"
    if job["method"] == "ours":
        predictors, _, _ = ours_predictor(job, srf, device)
        # Always use the full nonregistration pipeline, also at alpha=0.
        predict = predictors["warp"]
    else:
        from test_fixed_deformation import MODEL_BUILDERS
        cfg = model_args(job["dataset"], job["data_root"], str(device), job["seed"])
        cfg.mode, cfg.checkpoint, cfg.hidden_channels, cfg.eval_seed = "mixed", job["checkpoint"], None, job["eval_seed"]
        if job["method"] in SRF_CHECKED:
            report = checkpoint_srf_audit(job, cfg, srf)
            audit_status = report["status"]
            if audit_status not in ("exact", "roundoff"):
                raise ValueError(f"SRF audit {audit_status}; see SRF_AUDIT")
            cfg._verified_shared_srf = (report["checkpoint_sha256"], report["shared_sha256"])
        test, predict, _, _, _ = MODEL_BUILDERS[job["method"]](cfg)
        if not np.array_equal(np.asarray(test.srf_weights), srf):
            raise ValueError("Model SRF differs from the common observation SRF")
    devices = [] if device.type != "cuda" else [device.index if device.index is not None else torch.cuda.current_device()]
    for level, alpha, path in pending:
        totals = np.zeros((job["cases"], len(METRICS)), dtype=np.float64)
        for index, sample_path in enumerate(manifest["paths"]):
            with np.load(sample_path, allow_pickle=False) as sample:
                gt = torch.from_numpy(sample["gt"])[None].to(device)
                msi = torch.from_numpy(sample["hr_msi"])[None].to(device)
                observations = sample["observations"][level]
            for case in range(job["cases"]):
                reconstruction_seed = (job["eval_seed"] + index * 1000003 + case * 9176) % (2**63 - 1)
                with torch.no_grad(), torch.random.fork_rng(devices=devices):
                    torch.random.default_generator.manual_seed(reconstruction_seed)
                    if devices:
                        with torch.cuda.device(device):
                            torch.cuda.manual_seed(reconstruction_seed)
                    pred = predict(torch.from_numpy(observations[case])[None].to(device), msi)
                if pred.shape != gt.shape or not bool(torch.isfinite(pred).all()):
                    raise ValueError("Prediction must be finite with the GT shape")
                metrics = metric_fn(pred, gt, 4)
                if not all(math.isfinite(metrics[k]) for k in METRICS):
                    raise FloatingPointError("Nonfinite metric")
                totals[case] += [metrics[k] for k in METRICS]
            if (index + 1) % 16 == 0 or index + 1 == manifest["patches"]:
                print(f"PROGRESS {job['dataset']}/{job['method']} alpha={alpha:.2f} patches={index+1}/{manifest['patches']}", flush=True)
        case_means = totals / manifest["patches"]
        row = dict(dataset=job["dataset"], method=job["method"], strength=alpha,
                   translation_max_hr_px=4 * alpha, rotation_max_deg=2 * alpha, local_max_hr_px=4 * alpha,
                   status="ok", test_patches=manifest["patches"], cases_per_patch=job["cases"],
                   evaluations=manifest["patches"] * job["cases"], seed=job["seed"], eval_seed=job["eval_seed"],
                   input_sha256=manifest["input_sha256"], weights=json.dumps(weights, ensure_ascii=False),
                   srf_audit=audit_status, protocol=PROTOCOL, error="")
        row.update(zip(METRICS, case_means.mean(axis=0).tolist()))
        row.update(zip((k + "_std" for k in METRICS), case_means.std(axis=0).tolist()))
        write_json(path, dict(identity=identity, row=row, case_means=case_means.tolist(),
                              input_protocol=manifest["protocol"]))
        completed.append(row)
        write_json(Path(job["worker_rows"]), completed)
        print(f"RESULT alpha={alpha:.2f} PSNR={row['PSNR']:.6f} SAM={row['SAM']:.6f}", flush=True)


def save_csv(path, rows):
    temporary = path.with_suffix(".tmp.csv")
    with temporary.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for key in sorted(rows):
            writer.writerow({k: rows[key].get(k, "") for k in FIELDS})
    temporary.replace(path)


def parser(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    p.add_argument("--weights_root", type=Path, default=ROOT.parent / "模型权重")
    p.add_argument("--s2diff_root", type=Path, default=ROOT.parent / "S2Diff-MH")
    p.add_argument("--data_root", type=Path)
    p.add_argument("--output_dir", type=Path, default=ROOT / "outputs/warp_strength")
    p.add_argument("--ours_config", type=Path)
    p.add_argument("--checkpoint", action="append", default=[], metavar="METHOD:DATASET=PATH")
    p.add_argument("--training_srf", action="append", default=[], metavar="METHOD:DATASET=PATH")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    if min(args.cases, args.cpu_threads) < 1 or min(args.seed, args.eval_seed) < 0:
        p.error("cases/threads must be positive; seeds nonnegative")
    return args


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--_worker":
        job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        import_root = job["s2diff_root"] if job["method"] == "ours" else str(ROOT)
        sys.path.insert(0, import_root)
        os.chdir(import_root)
        worker(job)
        return 0
    args = parser(argv)
    for key in ("weights_root", "s2diff_root", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    if args.data_root:
        args.data_root = args.data_root.expanduser().resolve()
    from export_paper_predictions import checkpoint_for, ours_paths, worker_error
    from config import get_dataset_configs

    def data_root(dataset):
        if args.data_root:
            return args.data_root
        marker = get_dataset_configs()[dataset].file_name
        roots = (ROOT / "data/raw", args.s2diff_root / "data/raw")
        return next((root for root in roots if (root / marker).exists()), roots[0])

    def overrides(entries, allowed):
        result = {}
        for entry in entries:
            key, eq, value = entry.partition("=")
            method, sep, dataset = key.partition(":")
            if not eq or not sep or method not in allowed or dataset not in DATASETS or key in result:
                raise ValueError(f"Invalid override: {entry}")
            path = Path(value).expanduser().resolve()
            if not value or not path.is_file():
                raise FileNotFoundError(path)
            result[key] = str(path)
        return result

    checkpoints = overrides(args.checkpoint, METHODS[:-1])
    srfs = overrides(args.training_srf, ("M2DTN", "PMI-RFCoNet", "SA-RFM"))
    ours = json.loads(args.ours_config.read_text(encoding="utf-8")) if args.ours_config else {}
    jobs = []
    for dataset in dict.fromkeys(args.datasets):
        for method in dict.fromkeys(args.methods):
            job = dict(dataset=dataset, method=method, group="unregistered", device=args.device,
                       data_root=str(data_root(dataset)), weights_root=str(args.weights_root),
                       s2diff_root=str(args.s2diff_root), output_dir=str(args.output_dir),
                       cases=args.cases, seed=args.seed, eval_seed=args.eval_seed, cpu_threads=args.cpu_threads,
                       resume=args.resume, training_srf=srfs.get(f"{method}:{dataset}"),
                       ours_config=ours.get(dataset, {}).get("unregistered", {}))
            error = None
            try:
                if method == "ours":
                    if not (args.s2diff_root / "test_fixed_deformation.py").is_file():
                        raise FileNotFoundError(f"Update S2Diff-MH checkout: {args.s2diff_root}")
                    paths, _, _ = ours_paths(job)
                    job["planned_weights"] = {role: str(path) for role, path in paths.items()}
                else:
                    job["checkpoint"] = str(checkpoint_for(args.weights_root, method, "hsi_warp_final", dataset,
                                                          None, checkpoints.get(f"{method}:{dataset}")))
            except (FileNotFoundError, ValueError) as exc:
                error = str(exc)
            print(f"PLAN {dataset}/{method}: {error or job.get('checkpoint', 'ours mixed checkpoints')}", flush=True)
            jobs.append((job, error))
    if args.dry_run:
        return int(any(error for _, error in jobs))
    import torch
    torch.set_num_threads(args.cpu_threads)
    os.chdir(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "summary.csv"
    rows = {}
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8-sig") as handle:
            for row in csv.DictReader(handle):
                rows[(row["dataset"], row["method"], float(row["strength"]))] = row
    failed = False
    for dataset in dict.fromkeys(args.datasets):
        # Only this dataset's shared observations occupy disk, then are released.
        with TemporaryDirectory(prefix=f"shared-{dataset}-", dir=args.output_dir) as tmp:
            temp, manifest, input_error = Path(tmp), None, None
            for job, initial_error in [pair for pair in jobs if pair[0]["dataset"] == dataset]:
                error, worker_rows = initial_error, []
                if not error and manifest is None and input_error is None:
                    try:
                        manifest = prepare_inputs(job, temp / "inputs")
                    except Exception as exc:
                        input_error = f"{type(exc).__name__}: {exc}"
                error = error or input_error
                if not error:
                    job.update(manifest=str(temp / "inputs/manifest.json"),
                               result_dir=str(args.output_dir / "results" / dataset / job["method"]),
                               worker_rows=str(temp / "worker_rows.json"))
                    Path(job["worker_rows"]).unlink(missing_ok=True)
                    write_json(temp / "job.json", job)
                    log = args.output_dir / "logs" / dataset / f"{job['method']}.log"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    env = {**os.environ, "TORCHDYNAMO_DISABLE": "1"} if job["method"] == "M2DTN" else None
                    print(f"RUN {dataset}/{job['method']}; log={log}", flush=True)
                    with log.open("w", encoding="utf-8") as handle:
                        code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(temp / "job.json")],
                                              env=env, stdout=handle, stderr=subprocess.STDOUT).returncode
                    if Path(job["worker_rows"]).is_file():
                        worker_rows = json.loads(Path(job["worker_rows"]).read_text(encoding="utf-8"))
                    if code:
                        error = worker_error(log, code)
                available = {float(row["strength"]): row for row in worker_rows}
                for alpha in STRENGTHS:
                    row = available.get(alpha) or dict(dataset=dataset, method=job["method"], strength=alpha,
                        translation_max_hr_px=4*alpha, rotation_max_deg=2*alpha, local_max_hr_px=4*alpha,
                        status="failed", protocol=PROTOCOL, error=error or "Missing strength result")
                    rows[(dataset, job["method"], alpha)] = row
                save_csv(csv_path, rows)
                failed |= bool(error) or len(available) != len(STRENGTHS)
                print(f"{'FAILED' if error else 'OK'} {dataset}/{job['method']} {error or ''}", flush=True)
    print(f"SUMMARY {csv_path}", flush=True)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
