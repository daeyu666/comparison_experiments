"""Render archived native Augsburg-2 reconstructions without model inference."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
METHODS = ("CNMF", "HySure", "MoG-DCN", "PSRT", "ESFS", "CLSNet", "EMR-Diff",
           "M2DTN", "PMI-RFCoNet", "MoE-PNP", "SA-RFM", "ours")
CACHE_NAME = "augsburg2_wald_center_holdout"


def canonical(name):
    token = re.sub(r"[^a-z0-9]", "", name.lower())
    aliases = {re.sub(r"[^a-z0-9]", "", m.lower()): m for m in METHODS}
    aliases.update(s2diff="ours", s2diffmh="ours")
    if token not in aliases:
        raise ValueError(f"Unknown method {name!r}; supported: {', '.join(METHODS)}")
    return aliases[token]


def original_renderer():
    path = ROOT / "comparison/EMR-Diff/visualize_augsburg2_wald_center_holdout.py"
    spec = importlib.util.spec_from_file_location("original_augsburg_visualizer", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def file_sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_cube(results_root, method):
    directories = []
    if not results_root.is_dir():
        raise FileNotFoundError(results_root)
    for path in sorted(results_root.iterdir()):
        if not path.is_dir():
            continue
        try:
            if canonical(path.name) == method:
                directories.append(path)
        except ValueError:
            continue
    if len(directories) != 1:
        raise ValueError(f"Expected one {method} folder under {results_root}; found {len(directories)}. "
                         "Use --method NAME PATH to specify its saved cube.")
    variant = "base128" if method == "EMR-Diff" else "default"
    directory = directories[0] / variant
    standard = directory / "fused_hsi.npy"
    if standard.is_file():
        return standard.resolve(), variant
    candidates = sorted(p for p in directory.glob("*_HSI.npy") if p.is_file())
    if len(candidates) != 1:
        raise FileNotFoundError(f"Expected fused_hsi.npy or a unique *_HSI.npy in {directory}")
    return candidates[0].resolve(), variant


def reconstruction_rgb(path, expected_shape, bands):
    cube = np.load(path, mmap_mode="r", allow_pickle=False)
    if cube.ndim == 4 and cube.shape[0] == 1:
        cube = cube[0]
    layout = "HWC"
    if cube.shape != expected_shape:
        if cube.shape == (expected_shape[2], *expected_shape[:2]):
            cube = cube.transpose(1, 2, 0)
            layout = "CHW"
        else:
            raise ValueError(f"Expected heldout {expected_shape} HWC or CHW, got {cube.shape}: {path}")
    if cube.dtype.kind not in "fiu" or not np.isfinite(cube).all():
        raise ValueError(f"Reconstruction must be finite and numeric: {path}")
    return np.asarray(cube[..., list(bands)], dtype=np.float32), layout


def save_if_changed(image, path):
    """Keep existing references untouched when rerunning with the same data."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        with Image.open(path) as previous:
            if previous.mode == image.mode and previous.size == image.size and previous.tobytes() == image.tobytes():
                return
    image.save(path)


def contact_sheet(images, path, columns):
    tile, header, gap = 288, 28, 12
    columns = min(columns, len(images))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (columns * (tile + gap) + gap, rows * (tile + header + gap) + gap), "white")
    painter = ImageDraw.Draw(canvas)
    for i, (name, image) in enumerate(images):
        x = gap + (i % columns) * (tile + gap)
        y = gap + (i // columns) * (tile + header + gap)
        panel = image.copy()
        panel.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        canvas.paste(panel, (x + (tile - panel.width) // 2, y))
        label = "Ours" if name == "ours" else name
        box = painter.textbbox((0, 0), label)
        painter.text((x + (tile - (box[2] - box[0])) // 2, y + tile + 7), label, fill="black")
    save_if_changed(canvas, path)


def resolve_wald_root(explicit=None):
    if explicit:
        return explicit.expanduser().resolve()
    roots = (ROOT / "data" / CACHE_NAME, ROOT / "data", ROOT.parent / "S2Diff-MH/data" / CACHE_NAME)
    return next((p for p in roots if (p / "roi.json").is_file() and (p / "full/hr_msi.npy").is_file()), roots[0])


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_root", type=Path, default=ROOT.parent / "Real_world推理结果" / CACHE_NAME)
    p.add_argument("--wald_root", type=Path)
    p.add_argument("--output_dir", type=Path, default=ROOT / "outputs" / CACHE_NAME / "visualization_all")
    p.add_argument("--methods", nargs="+", type=canonical, default=METHODS)
    p.add_argument("--method", nargs=2, action="append", default=[], metavar=("NAME", "FUSED_NPY"),
                   help="Override an archived cube path; repeat for different methods")
    p.add_argument("--hsi_rgb", default="43,28,10", help="Zero-based R,G,B band indices, same as the original script")
    p.add_argument("--columns", type=int, default=4)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args(argv)
    bands = tuple(int(x) for x in args.hsi_rgb.split(","))
    if len(bands) != 3 or any(x < 0 or x >= 242 for x in bands) or args.columns < 1:
        p.error("Require three HSI band indices in [0,241] and columns >=1")
    results_root = args.results_root.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    if output == results_root or results_root in output.parents:
        p.error("Keep --output_dir outside the archived --results_root")
    overrides = {}
    for name, value in args.method:
        name = canonical(name)
        if name in overrides:
            p.error(f"Duplicate --method {name}")
        overrides[name] = Path(value).expanduser().resolve()
    selected = list(dict.fromkeys(args.methods))
    if set(overrides) - set(selected):
        p.error("Each --method override must also be in the selected --methods")
    plans, failures = [], []
    for name in selected:
        variant = "base128" if name == "EMR-Diff" else "default"
        try:
            path, variant = ((overrides[name], variant) if name in overrides else find_cube(results_root, name))
            if not path.is_file():
                raise FileNotFoundError(path)
            print(f"PLAN {name}/{variant}: {path}", flush=True)
            plans.append((name, variant, path))
        except (FileNotFoundError, ValueError) as exc:
            failures.append(dict(method=name, variant=variant, status="failed", error=str(exc)))
            print(f"FAILED {name}: {exc}", flush=True)
    if args.dry_run:
        return int(bool(failures))
    legacy = original_renderer()
    wald = resolve_wald_root(args.wald_root)
    roi = legacy.read_roi(wald)
    if roi is None or roi.get("source_region") != "sub_area_2":
        raise ValueError(f"Expected Region-2 center holdout roi.json under {wald}")
    y0, x0, y1, x1 = map(int, roi["test_bbox_10m"])
    expected = (y1 - y0, x1 - x0, 242)
    msi_path = wald / "full/hr_msi.npy"
    full_msi = np.load(msi_path, mmap_mode="r", allow_pickle=False)
    if (full_msi.ndim != 3 or full_msi.shape[-1] != 4 or not np.isfinite(full_msi).all()
            or not (0 <= y0 < y1 <= full_msi.shape[0] and 0 <= x0 < x1 <= full_msi.shape[1])):
        raise ValueError("Invalid four-band observed MSI or heldout bbox")
    full_rgb = np.asarray(full_msi[..., [2, 1, 0]], dtype=np.float32)
    overview = legacy.stretch(full_rgb)
    ImageDraw.Draw(overview).rectangle((x0, y0, x1-1, y1-1), outline=(255, 0, 0), width=3)
    overview_path = output / "Augsburg2_original_MSI_redbox.png"
    msi_roi_path = output / "Augsburg2_original_MSI_heldout_ROI.png"
    save_if_changed(overview, overview_path)
    save_if_changed(legacy.stretch(full_rgb[y0:y1, x0:x1]), msi_roi_path)
    rgbs, sources = {}, {}
    for name, variant, path in plans:
        try:
            rgb, layout = reconstruction_rgb(path, expected, bands)
            rgbs[name] = rgb
            sources[name] = dict(variant=variant, source=str(path), sha256=file_sha256(path), layout=layout)
        except (OSError, ValueError) as exc:
            failures.append(dict(method=name, variant=variant, status="failed", error=str(exc)))
            print(f"FAILED {name}: {exc}", flush=True)
    rows, images = [], []
    bounds = None
    if rgbs:
        # Match the original script: pool all methods before taking percentiles.
        values = np.concatenate([rgb.reshape(-1, 3) for rgb in rgbs.values()], axis=0)
        bounds = np.percentile(values, [1, 99], axis=0)
        for name, rgb in rgbs.items():
            image = legacy.stretch(rgb, bounds)
            dest = output / name / sources[name]["variant"] / "reconstruction_rgb.png"
            save_if_changed(image, dest)
            images.append((name, image))
            row = dict(method=name, **sources[name], status="ok", rgb=str(dest), error="")
            rows.append(row)
            print(f"RGB {name}: {dest}", flush=True)
        contact_sheet(images, output / "Augsburg2_all_methods_RGB.png", args.columns)
    rows.extend(failures)
    provenance = dict(protocol_id=roi["protocol_id"], source_region=roi["source_region"],
                      bbox_10m=roi["test_bbox_10m"], bbox_30m=roi["test_bbox_30m"],
                      results_root=str(results_root), wald_root=str(wald),
                      original_msi_sha256=file_sha256(msi_path), hsi_rgb_indices_0based=list(bands),
                      reconstruction_shared_percentiles=[1, 99],
                      shared_bounds=None if bounds is None else bounds.tolist(),
                      overview=str(overview_path), observed_msi_roi=str(msi_roi_path), results=rows,
                      inference_triggered_by_visualization=False, metrics_unchanged_by_visualization=True)
    (output / "visualization_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    with (output / "visualization_summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=("method", "variant", "status", "source", "sha256", "layout", "rgb", "error"))
        writer.writeheader()
        writer.writerows(rows)
    print(f"RENDERED {len(images)}/{len(selected)}; OVERVIEW {overview_path}", flush=True)
    return int(bool(failures) or not images)


if __name__ == "__main__":
    raise SystemExit(main())
