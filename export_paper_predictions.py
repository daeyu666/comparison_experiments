"""Export PaviaU/CAVE inference cubes and numeric error maps for paper figures."""
from __future__ import annotations

import argparse
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import runpy
import subprocess
import sys
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parent
DATASETS = ("PaviaU", "CAVE")
REGISTERED = ("CNMF", "HySure", "MoG-DCN", "PSRT", "ESFS", "CLSNet", "EMR-Diff", "ours")
UNREGISTERED = ("M2DTN", "PMI-RFCoNet", "MoE-PNP", "SA-RFM", "UAFL", "ours")
GROUPS = {"registered": REGISTERED, "unregistered": UNREGISTERED}
CLASSICAL = ("CNMF", "HySure")
SIZE = 128
ARRAYS = ("prediction.npy", "residual_rmse.npy", "sam_deg.npy")


def write_json(path, obj):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checkpoint_for(root, method, folder, dataset, width, override=None):
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(path)
        return path
    directory = root / f"{method}-checkpoints" / folder / dataset
    suffixes = (".pth.tar", ".pth", ".pt", ".ckpt")
    stem = "best" if method != "EMR-Diff" or width in (None, 64) else f"best_base{width}"
    specific = [stem]
    if method == "EMR-Diff" and width is None:
        specific.insert(0, "best_base128")
    elif method == "EMR-Diff" and width == 64:
        specific.append("best_base64")
    for name in specific:
        for ext in suffixes:
            candidate = directory / (name + ext)
            if candidate.is_file():
                return candidate.resolve()
    if width is None and directory.is_dir():
        choices = [p for p in sorted(directory.iterdir()) if p.is_file() and p.name.endswith(suffixes)
                   and not any(x in p.stem.lower() for x in ("last", "latest"))]
        if len(choices) == 1:
            return choices[0].resolve()
        if choices:
            raise ValueError(f"Ambiguous weights in {directory}; use --checkpoint METHOD:GROUP:DATASET=PATH")
    raise FileNotFoundError(f"No {stem} checkpoint in {directory}")


def cave_samples(test, crop, scenes):
    from data_loader import CAVE_TEST_SCENES

    selected = set(scenes or CAVE_TEST_SCENES)
    if not selected <= set(CAVE_TEST_SCENES):
        raise ValueError(f"Not held-out CAVE scenes: {sorted(selected - set(CAVE_TEST_SCENES))}")
    grid = [x for x in test.samples if x[0] in selected]
    if crop == "center":
        return [(name, (512 - SIZE) // 2, (512 - SIZE) // 2, SIZE)
                for name in CAVE_TEST_SCENES if name in selected]
    if crop == "central-grid":
        # The four central grid tiles are tied; choose the upper-left deterministically.
        return [x for x in grid if x[1:3] == (SIZE, SIZE)]
    return grid


def model_args(dataset, data_root, device, seed):
    from types import SimpleNamespace
    from benchmark_paviau import test_args

    cfg = test_args(SimpleNamespace(data_root=data_root, device=device, seed=seed))
    cfg.dataset = dataset
    return cfg


def dataset_and_info(cfg):
    from comparison.UAFL.common import build_shared_cfg
    from data_loader import build_datasets

    _, _, test, info = build_datasets(build_shared_cfg(cfg), include_validation=True)
    return test, info


def save_inputs(args, dataset, data_root):
    import numpy as np
    import torch
    from fixed_deformation_protocol import STRENGTHS, sample_fixed_geometry, tensor_sha256, warp_hsi

    test, info = dataset_and_info(model_args(dataset, data_root, "cpu", args.seed))
    if dataset == "CAVE":
        test = copy.copy(test)
        test.samples = cave_samples(test, args.cave_crop, args.cave_scenes)
        crops = test.samples
    else:
        if len(test) != 1:
            raise ValueError("Expected one official held-out PaviaU patch")
        crops = [("center", *test.coords[0], SIZE)]
    dest = args.output_dir / "raw" / dataset / "inputs"
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for idx, (scene, top, left, size) in enumerate(crops):
        sample = "center" if dataset == "PaviaU" else f"{scene}__r{top:03}_c{left:03}"
        item = test[idx]
        gt = item["gt"].unsqueeze(0)
        srf = torch.as_tensor(info["srf_weights"], dtype=torch.float32)
        # Seed by crop identity, independent of method and subset traversal order.
        crop_index = int.from_bytes(hashlib.sha256(f"{dataset}/{sample}".encode()).digest()[:4], "little")
        geom, record = sample_fixed_geometry(size, size, STRENGTHS[4], sample_index=crop_index,
                                             case_index=0, seed=args.warp_seed, exact_local_max=True)
        with torch.no_grad():
            lr_warp = test.degradation_operator.degrade(warp_hsi(gt, geom))[0]
        tensors = dict(gt=item["gt"], hr_msi=item["hr_msi"], lr_registered=item["lr_hsi"],
                       lr_warp=lr_warp, srf=srf, local_xy_hr_px=geom.local_field[0])
        crop_type = ("official-center" if dataset == "PaviaU" else
                     "center" if args.cave_crop == "center" else "official-grid")
        meta = dict(dataset=dataset, sample=sample, scene=scene, crop_rc_hw=[top, left, size, size],
                    crop_type=crop_type,
                    scale=4, degradation="physical", mtf_nyquist=0.2, psf_truncate=3.0,
                    wavelengths_nm=np.asarray(info["hsi_wavelengths"], dtype=float).tolist(),
                    warp_seed=args.warp_seed, warp=record,
                    tensor_sha256={k: tensor_sha256(v) for k, v in tensors.items()})
        path = dest / (sample + ".npz")
        if path.exists():
            with np.load(path, allow_pickle=False) as saved:
                old = json.loads(saved["metadata"].item())
            if old != meta:
                raise ValueError(f"Existing inputs differ: {path}. Use a different --output_dir.")
        else:
            tmp = path.with_suffix(".tmp.npz")
            np.savez_compressed(tmp, metadata=json.dumps(meta), **{k: v.numpy() for k, v in tensors.items()})
            tmp.replace(path)
        paths.append(str(path))
    return paths


def error_maps(prediction, target):
    """CHW -> pixelwise spectral RMSE and masked SAM in degrees."""
    import numpy as np

    p, t = (np.asarray(x, dtype=np.float64).clip(0, 1) for x in (prediction, target))
    if p.shape != t.shape or p.ndim != 3 or not np.isfinite(p).all() or not np.isfinite(t).all():
        raise ValueError("Predictions and ground truth must be finite and have the same CHW shape")
    residual = np.sqrt(np.mean((p - t) ** 2, axis=0))
    pn, tn = np.linalg.norm(p, axis=0), np.linalg.norm(t, axis=0)
    valid = (pn > 1e-12) & (tn > 1e-12)
    sam = np.full(residual.shape, np.nan, dtype=np.float32)
    cos = np.sum(p * t, axis=0)[valid] / (pn[valid] * tn[valid])
    sam[valid] = np.rad2deg(np.arccos(np.clip(cos, -1, 1)))
    return residual.astype(np.float32), sam


def solver(method, seed):
    from comparison.traditional.operators import PhysicalOperator
    from comparison.traditional.options import DEFAULTS as opts
    from threadpoolctl import threadpool_limits

    module = importlib.import_module(f"comparison.{method}.model")
    if method == "CNMF":
        cfg = module.CNMFConfig(opts["cnmf_iterations"], opts["cnmf_endmembers"], seed)
    else:
        cfg = module.HySureConfig(iterations=opts["hysure_iterations"], rank=opts["hysure_rank"],
                                  lambda_phi=opts["hysure_lambda_phi"], lambda_m=opts["hysure_lambda_m"],
                                  cg_maxiter=opts["hysure_cg_maxiter"], cg_rtol=opts["hysure_cg_rtol"])
    operator = PhysicalOperator(4, 0.2, 3.0)

    def predict(lr, hr, srf):
        with threadpool_limits(limits=opts["classical_threads"]):
            return module.fuse(lr, hr, srf.astype("float64"), operator, cfg)[0]
    return predict


def ours_paths(job):
    """Find explicit role weights; no substitution between training stages."""
    import torch

    root = Path(job["weights_root"])
    base = next((root / name for name in ("S2Diff-MH-checkpoints", "ours-checkpoints")
                 if (root / name).is_dir()), root / "S2Diff-MH-checkpoints")
    folder = "physical" if job["group"] == "registered" else "hsi_warp_final"
    directory = base / folder / job["dataset"]
    config = job.get("ours_config", {})
    configured = config.get("weights", {})
    files = sorted(p for p in directory.glob("*") if p.is_file() and p.name.endswith((".pth", ".pt", ".pth.tar"))
                   and "last" not in p.stem.lower())

    def role_path(role, recorded=None):
        if role in configured:
            path = Path(configured[role])
            if not path.is_absolute():
                path = root / path
            if not path.is_file():
                raise FileNotFoundError(path)
            return path.resolve()
        for ext in (".pth", ".pth.tar", ".pt"):
            if (directory / (role + ext)).is_file():
                return (directory / (role + ext)).resolve()
        if recorded:
            moved = directory / Path(recorded).name
            if moved.is_file():
                return moved.resolve()
        clues = {"refiner": ("refiner", "gigi"), "geometry": ("geometry", "recursive"),
                 "diffusion": ("diffusion", "raw_direct")}[role]
        matches = [p for p in files if any(c in p.name.lower() for c in clues)]
        if len(matches) == 1:
            return matches[0].resolve()
        raise ValueError(f"Cannot uniquely find ours {role} in {directory}; set weights.{role} in --ours_config")

    paths = {"refiner": role_path("refiner")}
    state = torch.load(paths["refiner"], map_location="cpu", weights_only=False)
    extra = state.get("extra", {})
    if extra.get("dataset", job["dataset"]) != job["dataset"]:
        raise ValueError("ours refiner checkpoint has another dataset")
    registered = job["group"] == "registered"
    paths["diffusion"] = role_path("diffusion", extra.get("baseline_checkpoint" if registered else "diffusion_checkpoint"))
    if not registered:
        paths["geometry"] = role_path("geometry", extra.get("geometry_checkpoint"))
    return paths, extra, config.get("options", {})


def ours_predictor(job, srf, device):
    import torch
    import train_innovation3_gigi_cdrdi as mixed

    registered = job["group"] == "registered"
    args = mixed.build_arg_parser(fixed_suite=True).parse_args([])
    args.dataset, args.stage, args.device, args.seed = job["dataset"], "test", str(device), job["seed"]
    paths, extra, configured = ours_paths(job)
    opts = ["diffusion_steps", "time_dim", "spectral_hidden", "dropout", "refine_hidden", "heads", "disable_tangent"]
    if registered:
        args.base_channels, args.baseline_checkpoint_type = 64, "standard"
        opts += ["base_channels", "baseline_checkpoint_type"]
    else:
        opts += ["geometry_steps", "geometry_base_channels", "diffusion_base_channels"]
    if set(configured) - set(opts):
        raise ValueError(f"Unsupported ours options: {sorted(set(configured) - set(opts))}")
    for key in opts:
        if key in configured or key in extra:
            setattr(args, key, configured.get(key, extra.get(key)))
    args.refiner_checkpoint = str(paths["refiner"])
    info = {"n_bands": srf.shape[1], "n_msi_bands": srf.shape[0], "srf_weights": srf}
    if registered:
        import train_innovation3_gigi as reg

        args.baseline_checkpoint = str(paths["diffusion"])
        base = reg._build_baseline(args, info, device)
        projector = reg._build_projector(args, info, device)
        refiner = reg._build_refiner(args, info, device).eval()
        reg._load_refiner_checkpoint(refiner, args.refiner_checkpoint, device)
        process = reg.build_progressive_process(reg._config(args))

        def predict(lr, msi):
            pred = reg.reconstruct_from_terminal_lr(base, process, lr,
                                                   target_size=tuple(msi.shape[-2:]), hr_msi=msi)
            residual = reg._physical_residual(process, projector, lr, pred)
            return refiner(pred, msi, residual, reg.ranked_msi_heterogeneity(msi))

        predictors = {"registered": predict}
    else:
        args.geometry_checkpoint = str(paths["geometry"])
        args.diffusion_checkpoint = str(paths["diffusion"])
        process = mixed.build_progressive_process(mixed._config(args))
        build = runpy.run_path(str(Path.cwd() / "test_fixed_deformation.py"))["build_predictor"]
        predict, predict_registered, _ = build(args, info, process, device, include_registered=True)
        predictors = {"registered": predict_registered, "warp": predict}
    return predictors, paths, {key: getattr(args, key) for key in opts}


def result_dir(root, dataset, group, condition, method, sample):
    path = root / "raw" / dataset / group
    if group == "unregistered":
        path /= condition
    return path / method / sample


def export_worker(job):
    import numpy as np
    import torch

    dataset, method, group = (job[key] for key in ("dataset", "method", "group"))
    device = torch.device("cpu" if method in CLASSICAL else job["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; pass --device cpu explicitly")
    torch.set_num_threads(job["cpu_threads"])
    torch.manual_seed(job["seed"])
    cfg_paths, model_options = {}, {}
    first = Path(job["inputs"][0])
    with np.load(first, allow_pickle=False) as initial:
        srf = initial["srf"]
    conditions = ("registered",) if group == "registered" else ("registered", "warp")
    if method in CLASSICAL:
        classical = solver(method, job["seed"])
    elif method == "ours":
        predictors, cfg_paths, model_options = ours_predictor(job, srf, device)
    else:
        cfg = model_args(dataset, job["data_root"], str(device), job["seed"])
        cfg.mode = "registered" if group == "registered" else "mixed"
        cfg.checkpoint = job["checkpoint"]
        cfg.hidden_channels = job["emr_width"] if method == "EMR-Diff" else None
        if method in ("MoG-DCN", "ESFS"):
            from comparison.registered_fusion.runner import load_state, make_model

            _, info = dataset_and_info(cfg)
            predict, _, expected = make_model(method, cfg, info)
            load_state(cfg.checkpoint, predict, expected)
            predict.eval()
            model_srf = info["srf_weights"]
        else:
            from test_fixed_deformation import MODEL_BUILDERS

            test, predict, _, _, _ = MODEL_BUILDERS[method](cfg)
            model_srf = test.srf_weights
        if not np.array_equal(np.asarray(model_srf), srf):
            raise ValueError(f"{method} test SRF differs from the shared input SRF")
        predictors = {c: predict for c in conditions}
        cfg_paths = {"model": Path(cfg.checkpoint)}

    weights = {key: dict(path=str(path), sha256=sha256(path)) for key, path in cfg_paths.items()}
    root = Path(job["output_dir"])
    done = 0
    with torch.no_grad():
        for path in job["inputs"]:
            with np.load(path, allow_pickle=False) as input_file:
                sample = json.loads(input_file["metadata"].item())
                gt, msi = input_file["gt"], input_file["hr_msi"]
                for condition in conditions:
                    dest = result_dir(root, dataset, group, condition, method, sample["sample"])
                    identity = dict(dataset=dataset, group=group, condition=condition, method=method,
                                    sample=sample["sample"], checkpoint=weights, model_options=model_options,
                                    seed=job["seed"], input_sha256=sample["tensor_sha256"])
                    meta_file = dest / "metadata.json"
                    if job["resume"] and meta_file.is_file() and all((dest / n).is_file() for n in ARRAYS):
                        if json.loads(meta_file.read_text(encoding="utf-8"))["identity"] != identity:
                            raise ValueError(f"Cannot resume with different inputs/weights: {dest}")
                        done += 1
                        continue
                    lr = input_file[f"lr_{condition}"]
                    # For stochastic methods, use paired random draws across conditions.
                    torch.manual_seed(job["seed"])
                    if method in CLASSICAL:
                        pred = classical(lr, msi, srf)
                    else:
                        pred = predictors[condition](torch.from_numpy(lr)[None].to(device),
                                                     torch.from_numpy(msi)[None].to(device))[0].detach().cpu().numpy()
                    pred = np.asarray(pred, dtype=np.float32)
                    if not np.isfinite(pred).all():
                        raise FloatingPointError(f"Nonfinite prediction: {dataset}/{method}/{condition}/{sample['sample']}")
                    residual, sam = error_maps(pred, gt)
                    dest.mkdir(parents=True, exist_ok=True)
                    # Publish metadata last so an interrupted sample is not resumed.
                    meta_file.unlink(missing_ok=True)
                    for name, array in zip(ARRAYS, (pred, residual, sam)):
                        final = dest / name
                        tmp = final.with_suffix(".tmp.npy")
                        np.save(tmp, array, allow_pickle=False)
                        tmp.replace(final)
                    write_json(meta_file, dict(identity=identity, input_file=str(Path(path).relative_to(root)),
                                               device=str(device), valid_sam_pixels=int(np.isfinite(sam).sum())))
                    done += 1
                    print(f"OK {dataset}/{group}/{condition}/{method}/{sample['sample']}", flush=True)
    print(f"SAVED {done} {dataset}/{group}/{method}", flush=True)


def parse_args(argv):
    from emr_configuration import validate_hidden_channels

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--groups", nargs="+", choices=tuple(GROUPS), default=tuple(GROUPS))
    p.add_argument("--methods", nargs="+", choices=sorted(set(REGISTERED + UNREGISTERED)))
    p.add_argument("--data_root", type=Path)
    p.add_argument("--weights_root", type=Path, default=ROOT.parent / "模型权重")
    p.add_argument("--s2diff_root", type=Path, default=ROOT.parent / "S2Diff-MH")
    p.add_argument("--output_dir", type=Path, default=ROOT / "outputs/paper_visualization")
    p.add_argument("--cave_crop", choices=("center", "central-grid", "all-grid"), default="center")
    p.add_argument("--cave_scenes", nargs="+")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cpu_threads", type=int, default=1)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--warp_seed", type=int, default=10)
    p.add_argument("--emr_width", type=validate_hidden_channels)
    p.add_argument("--checkpoint", action="append", default=[], metavar="METHOD:GROUP:DATASET=PATH")
    p.add_argument("--ours_config", type=Path, help="JSON with dataset/group weight and architecture overrides")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    if args.cpu_threads < 1 or min(args.seed, args.warp_seed) < 0:
        p.error("Threads must be positive and seeds must be nonnegative")
    return args


def selected_data_root(args, dataset):
    if args.data_root:
        return args.data_root.expanduser().resolve()
    roots = (ROOT / "data/raw", args.s2diff_root / "data/raw")
    marker = "PaviaU.mat" if dataset == "PaviaU" else "CAVE"
    return next((p for p in roots if (p / marker).exists()), roots[0])


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) == 2 and argv[0] == "--_worker":
        job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
        import_root = job["s2diff_root"] if job["method"] == "ours" else str(ROOT)
        sys.path.insert(0, import_root)
        os.chdir(import_root)
        export_worker(job)
        return 0

    args = parse_args(argv)
    for key in ("weights_root", "s2diff_root", "output_dir"):
        setattr(args, key, getattr(args, key).expanduser().resolve())
    overrides = {}
    for entry in args.checkpoint:
        key, eq, value = entry.partition("=")
        parts = key.split(":")
        if (not eq or not value or len(parts) != 3 or parts[1] not in GROUPS or parts[2] not in DATASETS
                or parts[0] not in GROUPS[parts[1]] or parts[0] in (*CLASSICAL, "ours") or key in overrides):
            raise ValueError(f"Invalid --checkpoint override: {entry}")
        overrides[key] = value
    ours_config = json.loads(args.ours_config.read_text(encoding="utf-8")) if args.ours_config else {}
    plans = []
    for dataset in dict.fromkeys(args.datasets):
        data_root = selected_data_root(args, dataset)
        for group in dict.fromkeys(args.groups):
            folder = "physical" if group == "registered" else "hsi_warp_final"
            for method in GROUPS[group]:
                if args.methods and method not in args.methods:
                    continue
                job = dict(dataset=dataset, group=group, method=method, data_root=str(data_root),
                           output_dir=str(args.output_dir), device=args.device, seed=args.seed,
                           emr_width=args.emr_width, cpu_threads=args.cpu_threads,
                           s2diff_root=str(args.s2diff_root), weights_root=str(args.weights_root),
                           resume=args.resume)
                error, description = None, "CPU solver"
                try:
                    if method == "ours":
                        if not (args.s2diff_root / "train_innovation3_gigi_cdrdi.py").is_file():
                            raise FileNotFoundError(f"S2Diff-MH checkout not found: {args.s2diff_root}")
                        job["ours_config"] = ours_config.get(dataset, {}).get(group, {})
                        description = "S2Diff-MH " + folder + " (separate checkpoint roles)"
                    elif method not in CLASSICAL:
                        override = overrides.get(f"{method}:{group}:{dataset}")
                        path = checkpoint_for(args.weights_root, method, folder, dataset,
                                              args.emr_width if method == "EMR-Diff" else None, override)
                        job["checkpoint"] = description = str(path)
                except (ValueError, FileNotFoundError) as exc:
                    error = str(exc)
                print(f"PLAN {dataset}/{group}/{method}: {error or description}", flush=True)
                plans.append((job, error))
    if args.dry_run:
        return int(any(error for _, error in plans))
    if not plans:
        raise SystemExit("No selected methods in the requested groups")

    import torch
    torch.set_num_threads(args.cpu_threads)
    os.chdir(ROOT)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests, inputs = [], {}
    with TemporaryDirectory(prefix="paper-visualization-") as temp:
        spec = Path(temp) / "job.json"
        for job, error in plans:
            dataset, group, method = (job[k] for k in ("dataset", "group", "method"))
            log = args.output_dir / "logs" / dataset / group / (method + ".log")
            if not error:
                if dataset not in inputs:
                    try:
                        inputs[dataset] = save_inputs(args, dataset, Path(job["data_root"])), None
                    except Exception as exc:
                        inputs[dataset] = None, f"{type(exc).__name__}: {exc}"
                job["inputs"], error = inputs[dataset]
            if not error:
                write_json(spec, job)
                log.parent.mkdir(parents=True, exist_ok=True)
                cwd = args.s2diff_root if method == "ours" else ROOT
                with log.open("w", encoding="utf-8") as stream:
                    code = subprocess.run([sys.executable, str(Path(__file__).resolve()), "--_worker", str(spec)],
                                          cwd=cwd, stdout=stream, stderr=subprocess.STDOUT).returncode
                if code:
                    lines = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()
                    error = f"{lines[-1] if lines else f'Worker exited with code {code}'}; see {log}"
            manifest = dict(dataset=dataset, group=group, method=method, status="failed" if error else "ok",
                            error=error, log=str(log) if log.is_file() else None)
            manifests.append(manifest)
            print(f"{manifest['status'].upper()} {dataset}/{group}/{method} {error or ''}", flush=True)
            write_json(args.output_dir / "export_summary.json", manifests)
    return int(any(m["status"] == "failed" for m in manifests))


if __name__ == "__main__":
    raise SystemExit(main())
