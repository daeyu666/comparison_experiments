"""Discover local Wald checkpoints and evaluate Augsburg Region 2 with native QNR."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

from emr_configuration import validate_hidden_channels
from experiment_protocol import (
    REAL_CHECKPOINT_FAMILY, REAL_CLASSICAL_METHODS, REAL_METHODS, REAL_RADIOMETRY,
    REAL_QNR_MIN_VALID_FRACTION, REAL_QNR_SUPPORT_FRACTION, REAL_QNR_WINDOW_HR,
    REAL_SPLIT_PROTOCOL, REAL_TEST_BBOX_10M, REAL_TILE_SIZE, REAL_TILE_STRIDE, REAL_WALD_ROOT,
    ROOT, build_real_native_test_command, build_real_reference_test_command, resolve_model,
)

_WEIGHT = re.compile(r"^(best|last)(?:_base([0-9]+))?\.(?:pth\.tar|pth|pt)$")


@dataclass(frozen=True)
class Job:
    method: str
    checkpoint: Path | None
    variant: str
    width: int | None = None


def method_checkpoint_dirs(root, folder):
    """Accept a repo root, a comparison folder, or a method/checkpoints/Wald folder."""
    candidates = [
        root / "comparison" / folder / "checkpoints" / REAL_CHECKPOINT_FAMILY,
        root / folder / "checkpoints" / REAL_CHECKPOINT_FAMILY,
    ]
    if folder in root.parts:
        if root.name == folder:
            candidates.append(root / "checkpoints" / REAL_CHECKPOINT_FAMILY)
        elif root.name == "checkpoints":
            candidates.append(root / REAL_CHECKPOINT_FAMILY)
        elif REAL_CHECKPOINT_FAMILY in root.parts:
            candidates.append(root)
    return [p for p in candidates if p.is_dir()]


def discover(method, roots, *, width=None, allow_last=False):
    adapter = resolve_model(method)
    seen = set()
    jobs = []
    for root in roots:
        for directory in method_checkpoint_dirs(root, adapter.folder):
            selected = {}
            for file in sorted(directory.rglob("*")):
                match = _WEIGHT.fullmatch(file.name)
                if not file.is_file() or not match:
                    continue
                kind, base = match.groups()
                if base and (method != "EMR-Diff" or int(base) < 32 or int(base) % 32):
                    continue
                name_width = int(base) if base else None
                if method == "EMR-Diff" and width is not None and (name_width or 64) != width:
                    continue
                if kind == "last" and not allow_last:
                    continue
                key = (file.parent, (name_width or 64) if method == "EMR-Diff" else None)
                # Prefer best, followed by the canonical .pth.tar extension.
                priority = (kind != "best", not file.name.endswith(".pth.tar"), file.name)
                if key not in selected or priority < selected[key][0]:
                    selected[key] = (priority, file, kind, name_width)
            for _, file, kind, name_width in sorted(selected.values(), key=lambda x: str(x[1])):
                path = file.resolve()
                if path in seen:
                    continue
                seen.add(path)
                variant = f"base{name_width or 64}" if method == "EMR-Diff" else "default"
                if kind == "last":
                    variant += "_last"
                jobs.append(Job(method, path, variant, name_width))
    if len(jobs) > 1:
        names = [job.variant for job in jobs]
        jobs = [
            Job(job.method, job.checkpoint,
                f"{job.variant}_{hashlib.sha256(str(job.checkpoint).encode()).hexdigest()[:8]}"
                if names.count(job.variant) > 1 else job.variant, job.width)
            for job in jobs
        ]
    return jobs


def _path(value):
    return Path(value).expanduser().resolve()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", nargs="+", default=["all"], help="One or more model names, or all")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--checkpoint", type=_path, help="Explicit real Wald checkpoint for one neural model")
    p.add_argument("--checkpoint_root", action="append", type=_path, default=[],
                   help="Extra local repo, comparison or checkpoint folder; repeat for separate disks")
    p.add_argument("--base_channels", "--base_channel", "--hidden_channels", "--model_width",
                   dest="width", type=validate_hidden_channels, help="Filter/validate the EMR-Diff base width")
    p.add_argument("--allow_last", action="store_true", help="Use last weights only when no best exists for that variant")
    p.add_argument("--wald_root", type=_path, default=(ROOT / REAL_WALD_ROOT).resolve())
    p.add_argument("--radiometry_json", type=_path, default=(ROOT / REAL_RADIOMETRY).resolve())
    p.add_argument("--output_dir", type=_path, default=ROOT / "outputs" / REAL_CHECKPOINT_FAMILY)
    p.add_argument("--tile_size", type=int, default=REAL_TILE_SIZE)
    p.add_argument("--tile_stride", type=int, default=REAL_TILE_STRIDE)
    p.add_argument("--no_geotiff", action="store_true")
    p.add_argument("--reference", action="store_true", help="Also run the separate 30m Wald reference test")
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    if "all" in args.model:
        if args.model != ["all"]:
            p.error("Use all by itself, or list specific methods")
        args.model = REAL_METHODS
    else:
        args.model = tuple(dict.fromkeys(resolve_model(name).canonical for name in args.model))
    if any(name not in REAL_METHODS for name in args.model):
        p.error("Select one of the Augsburg methods listed in the README")
    if args.width is not None and "EMR-Diff" not in args.model:
        p.error("Base width only applies to EMR-Diff")
    if args.checkpoint and (len(args.model) != 1 or args.model[0] in REAL_CLASSICAL_METHODS):
        p.error("--checkpoint requires one neural model")
    if not 0 < args.tile_stride <= args.tile_size <= 144 or any(n % 24 for n in (args.tile_size, args.tile_stride)):
        p.error("Tile and stride must be multiples of 24, with 0 < stride <= tile <= 144")
    return args


def plan(args):
    jobs, missing = [], []
    roots = [ROOT, *args.checkpoint_root]
    for method in args.model:
        if method in REAL_CLASSICAL_METHODS:
            jobs.append(Job(method, None, "solver"))
        elif args.checkpoint:
            if {"physical", "gaussian_bicubic", "hsi_warp_final"} & set(args.checkpoint.parts):
                raise ValueError("Synthetic x4 checkpoint paths are not accepted in the real x3 suite")
            match = _WEIGHT.fullmatch(args.checkpoint.name)
            named_width = int(match.group(2)) if match and match.group(2) and method == "EMR-Diff" else None
            if args.width and named_width and args.width != named_width:
                raise ValueError("Requested EMR-Diff width disagrees with checkpoint name")
            width = args.width or named_width
            kind = "_last" if match and match.group(1) == "last" else ""
            variant = (f"base{width}" if width else "explicit") + kind
            jobs.append(Job(method, args.checkpoint, variant, width))
        else:
            found = discover(method, roots, width=args.width if method == "EMR-Diff" else None,
                             allow_last=args.allow_last)
            jobs.extend(found)
            if not found:
                missing.append(method)
    return jobs, missing


def command(job, args, output, *, reference=False):
    kwargs = dict(
        checkpoint=job.checkpoint, hidden_channels=job.width, wald_root=args.wald_root,
        radiometry_json=args.radiometry_json, output=output, check_files=not args.dry_run,
    )
    adapter = resolve_model(job.method)
    if reference:
        return build_real_reference_test_command(adapter, args.device, **kwargs)
    return build_real_native_test_command(adapter, args.device, write_tif=not args.no_geotiff,
                                          tile_size=args.tile_size, tile_stride=args.tile_stride, **kwargs)


def normalize_outputs(folder, geotiff):
    files = {"qnr.json": "*_QNR.json", "fused_hsi.npy": "*_HSI.npy", "protocol.json": "*_protocol.json"}
    if geotiff:
        files["fused_hsi.tif"] = "*_HSI.tif"
    for name, pattern in files.items():
        target = folder / name
        if not target.is_file():
            matches = list(folder.glob(pattern))
            if len(matches) != 1:
                raise FileNotFoundError(f"Expected exactly one {name} output in {folder}")
            matches[0].replace(target)
    metrics = json.loads((folder / "qnr.json").read_text(encoding="utf-8"))
    required = {
        "split_protocol_id": REAL_SPLIT_PROTOCOL, "test_bbox_10m": list(REAL_TEST_BBOX_10M),
        "high_window": REAL_QNR_WINDOW_HR, "window_min_valid_fraction": REAL_QNR_MIN_VALID_FRACTION,
        "srf_support_fraction_of_peak": REAL_QNR_SUPPORT_FRACTION,
        "spectral_pair_count": 242 * 241 // 2, "srf_projection_used": False,
    }
    if any(metrics.get(k) != v for k, v in required.items()):
        raise ValueError("Worker metrics differ from the shared Wald ROI or QNR definition")
    for key, upper in (("QNR", 1), ("Dlambda", 2), ("Ds", 2)):
        if not isinstance(metrics[key], (int, float)) or not 0 <= metrics[key] <= upper:
            raise ValueError(f"Invalid {key}: {metrics[key]!r}")
    return metrics


def run_job(job, args, dest):
    with tempfile.TemporaryDirectory(prefix=f".{job.method}-", dir=args.output_dir) as temp:
        staging = Path(temp)
        cmd = command(job, args, staging)
        print("COMMAND " + shlex.join(cmd), flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        metrics = normalize_outputs(staging, not args.no_geotiff)
        if args.reference:
            metadata = json.loads((staging / "protocol.json").read_text(encoding="utf-8"))
            if job.method == "EMR-Diff":
                job = replace(job, width=metadata.get("backbone_width", job.width))
            ref = command(job, args, staging, reference=True)
            with (staging / "reference.log").open("w", encoding="utf-8") as handle:
                subprocess.run(ref, cwd=ROOT, check=True, stdout=handle, stderr=subprocess.STDOUT)
        dest.mkdir(parents=True, exist_ok=True)
        for file in staging.iterdir():
            file.replace(dest / file.name)
        return metrics


def write_summary(output, rows, args):
    output.mkdir(parents=True, exist_ok=True)
    info = {"protocol": REAL_SPLIT_PROTOCOL, "wald_root": str(args.wald_root),
            "radiometry_json": str(args.radiometry_json), "results": rows}
    path = output / "summary.json"
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(info, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=(
            "method", "variant", "status", "checkpoint", "backbone_width", "QNR", "Dlambda", "Ds",
            "output_dir", "error",
        ))
        writer.writeheader()
        writer.writerows(rows)


def main(argv=None):
    args = parse_args(argv)
    jobs, missing = plan(args)
    print(f"REAL_AUGSBURG_TEST {REAL_SPLIT_PROTOCOL}; native x3 QNR")
    rows = []
    for method in missing:
        print(f"MISSING_CHECKPOINT {method}", flush=True)
        rows.append(dict(method=method, status="missing_checkpoint"))
    if not args.dry_run and jobs:
        from comparison.real_augsburg.common import WaldCache
        WaldCache.load(args.wald_root, args.radiometry_json)
        args.output_dir.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        dest = args.output_dir / job.method / job.variant
        print(f"JOB {job.method}/{job.variant} checkpoint={job.checkpoint or 'solver'}", flush=True)
        if args.dry_run:
            print("COMMAND " + shlex.join(command(job, args, dest)))
            if args.reference:
                print("REFERENCE " + shlex.join(command(job, args, dest, reference=True)))
            continue
        row = dict(method=job.method, variant=job.variant, checkpoint=str(job.checkpoint or ""),
                   output_dir=str(dest))
        try:
            metrics = run_job(job, args, dest)
            protocol = json.loads((dest / "protocol.json").read_text(encoding="utf-8"))
            if protocol.get("backbone_width") is not None:
                row["backbone_width"] = protocol["backbone_width"]
            row.update(status="ok", **{k: metrics[k] for k in ("QNR", "Dlambda", "Ds")})
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
            row.update(status="failed", error=str(exc))
            print(f"FAILED {job.method}/{job.variant}: {exc}", flush=True)
        rows.append(row)
    if not args.dry_run:
        write_summary(args.output_dir, rows, args)
        print(f"WALD_BATCH_SUMMARY={args.output_dir / 'summary.csv'}", flush=True)
        return 1 if not jobs or any(row["status"] == "failed" for row in rows) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
