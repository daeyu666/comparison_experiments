"""Render saved paper predictions, residual RMSE maps and SAM maps as PNG."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent
DATASETS = ("PaviaU", "CAVE")
TARGETS = (("residual_rmse", 1.0, "Spectral RMSE"),
           ("sam_deg", 90.0, "SAM (deg)"))
ERROR_CMAP = "jet"
COLORBAR_ORIENTATION = "vertical"
DEFAULT_LIMIT_FACTORS = {("PaviaU", "residual_rmse"): 1.25}
DEFAULT_GAMMAS = {("PaviaU", "residual_rmse"): 3.0, ("PaviaU", "sam_deg"): 1.25}
DEFAULT = ROOT / "outputs/paper_visualization"


def find_results(raw, dataset):
    paths = [*(raw / dataset / "registered").glob("*/*/metadata.json"),
             *(raw / dataset / "unregistered").glob("*/*/*/metadata.json")]
    return [(path, json.loads(path.read_text(encoding="utf-8"))) for path in sorted(paths)]


def rgb_parameters(gt, wavelengths_nm, bands_nm):
    wavelengths = np.asarray(wavelengths_nm)
    if wavelengths.shape != (gt.shape[0],) or not np.isfinite(wavelengths).all():
        raise ValueError("Saved wavelength vector does not match the HSI cube")
    bands = [int(np.argmin(np.abs(wavelengths - nm))) for nm in bands_nm]
    lo, hi = np.percentile(gt[bands].clip(0, 1), [1, 99], axis=(1, 2))
    return bands, lo, np.maximum(hi, lo + 1e-6)


def to_rgb(cube, bands, low, high):
    stretched = (cube[bands].clip(0, 1) - low[:, None, None]) / (high - low)[:, None, None]
    return np.round(255 * stretched.clip(0, 1).transpose(1, 2, 0)).astype(np.uint8)


def common_limit(results, field, bound, percentile):
    # Fixed histogram covers all saved pixels without loading the full dataset into memory.
    edges = np.linspace(0, bound, 16385)
    counts = np.zeros(len(edges) - 1, dtype=np.int64)
    for meta, _ in results:
        values = np.load(meta.parent / (field + ".npy"), allow_pickle=False)
        good = np.isfinite(values)
        counts += np.histogram(values[good].clip(0, bound), bins=edges)[0]
    total = counts.sum()
    if not total:
        return bound
    index = int(np.searchsorted(np.cumsum(counts), np.ceil(total * percentile / 100)))
    return float(edges[index + 1])


def save_image(array, path, zoom):
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.fromarray(array)
    if zoom > 1:
        image = image.resize((image.width * zoom, image.height * zoom), Image.Resampling.NEAREST)
    image.save(path)


def map_rgb(values, cmap, norm):
    valid = np.isfinite(values)
    normalized = norm(np.where(valid, values, 0))
    colors = matplotlib.colormaps[cmap](normalized, bytes=True)[..., :3].copy()
    colors[~valid] = 180
    return colors


def save_colorbar(path, cmap, norm, label):
    fig = plt.figure(figsize=(1.5, 3.6), dpi=180)
    axis = fig.add_axes((0.13, 0.08, 0.13, 0.84))
    ticks = norm.inverse(np.linspace(0, 1, 5))
    bar = fig.colorbar(matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap),
                       cax=axis, orientation=COLORBAR_ORIENTATION,
                       ticks=ticks, format="%.3g")
    bar.set_label(label)
    fig.savefig(path, facecolor="white")
    plt.close(fig)


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, default=DEFAULT, help="Export directory containing raw/")
    p.add_argument("--output_dir", type=Path, help="Default: INPUT_DIR/figures")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=DATASETS)
    p.add_argument("--groups", nargs="+", choices=("registered", "unregistered"))
    p.add_argument("--methods", nargs="+")
    p.add_argument("--samples", nargs="+", help="Scene names or exact crop IDs, e.g. jelly_beans")
    p.add_argument("--rgb_nm", type=float, nargs=3, default=(650, 550, 470), metavar=("R", "G", "B"))
    p.add_argument("--residual_max", type=float, help="Override the pooled spectral RMSE scale")
    p.add_argument("--residual_gamma", type=float,
                   help="RMSE color scale exponent (default: PaviaU 3, CAVE 1; 1 is linear)")
    p.add_argument("--sam_max", type=float, help="Override the pooled SAM scale in degrees")
    p.add_argument("--sam_gamma", type=float,
                   help="SAM color scale exponent (default: PaviaU 1.25, CAVE 1; 1 is linear)")
    p.add_argument("--error_percentile", type=float, default=99.0)
    p.add_argument("--zoom", type=int, default=3)
    args = p.parse_args(argv)
    if not 0 < args.error_percentile <= 100 or args.zoom < 1:
        p.error("Require 0 < error_percentile <= 100 and zoom >= 1")
    for value in (args.residual_max, args.sam_max):
        if value is not None and not (np.isfinite(value) and value > 0):
            p.error("Error map maxima must be positive and finite")
    for field in ("residual_gamma", "sam_gamma"):
        value = getattr(args, field)
        if value is not None and not (np.isfinite(value) and value > 0):
            p.error(f"--{field} must be positive and finite")
    if not np.isfinite(args.rgb_nm).all():
        p.error("RGB wavelengths must be finite")
    return args


def main(argv=None):
    args = parse_args(argv)
    root = args.input_dir.expanduser().resolve()
    output = (args.output_dir or root / "figures").expanduser().resolve()
    raw = root / "raw"
    rendered = 0
    for dataset in dict.fromkeys(args.datasets):
        results = find_results(raw, dataset)
        if not results:
            print(f"No exported {dataset} results under {raw}")
            continue
        # Cache only small display settings, not the raw cubes.
        settings = {}
        limits = {"residual_rmse": args.residual_max, "sam_deg": args.sam_max}
        gammas = {"residual_rmse": args.residual_gamma, "sam_deg": args.sam_gamma}
        limit_factors = {}
        norms = {}
        for field, bound, label in TARGETS:
            if gammas[field] is None:
                gammas[field] = DEFAULT_GAMMAS.get((dataset, field), 1.0)
            if limits[field] is None:
                factor = DEFAULT_LIMIT_FACTORS.get((dataset, field), 1.0)
                limits[field] = min(bound, factor * common_limit(results, field, bound, args.error_percentile))
                limit_factors[field] = factor
            limits_arg = dict(vmin=0, vmax=limits[field], clip=True)
            norms[field] = (Normalize(**limits_arg) if gammas[field] == 1.0
                            else PowerNorm(gammas[field], **limits_arg))
            path = output / dataset / (field + "_colorbar.png")
            path.parent.mkdir(parents=True, exist_ok=True)
            save_colorbar(path, ERROR_CMAP, norms[field], label)
        notes = {"error_percentile": args.error_percentile, "error_limits": limits,
                 "auto_limit_factors": limit_factors, "error_gamma": gammas,
                 "error_colormap": ERROR_CMAP, "colorbar_orientation": COLORBAR_ORIENTATION,
                 "rgb_wavelength_targets_nm": list(args.rgb_nm), "invalid_sam_gray": 180,
                 "samples": {}}
        for meta, row in results:
            identity = row["identity"]
            if args.groups and identity["group"] not in args.groups:
                continue
            if args.methods and identity["method"] not in args.methods:
                continue
            input_path = (root / row["input_file"]).resolve()
            sample = identity["sample"]
            if sample not in settings:
                with np.load(input_path, allow_pickle=False) as data:
                    info = json.loads(data["metadata"].item())
                    gt = data["gt"]
                rgb = rgb_parameters(gt, info["wavelengths_nm"], args.rgb_nm)
                settings[sample] = (info, rgb)
            info, rgb = settings[sample]
            if args.samples and sample not in args.samples and info["scene"] not in args.samples:
                continue
            if info["tensor_sha256"] != identity["input_sha256"]:
                raise ValueError(f"Export references different observations: {meta}")
            if sample not in notes["samples"]:
                with np.load(input_path, allow_pickle=False) as data:
                    refs = output / dataset / "references" / sample
                    save_image(to_rgb(data["gt"], *rgb), refs / "gt_rgb.png", args.zoom)
                bands, low, high = rgb
                notes["samples"][sample] = {"scene": info["scene"], "rgb_band_indices_0based": bands,
                    "rgb_wavelengths_nm": [info["wavelengths_nm"][i] for i in bands],
                    "rgb_gt_percentiles": [low.tolist(), high.tolist()]}
            dest = output / meta.parent.relative_to(raw)
            cube = np.load(meta.parent / "prediction.npy", allow_pickle=False)
            save_image(to_rgb(cube, *rgb), dest / "prediction_rgb.png", args.zoom)
            for field, _, _ in TARGETS:
                values = np.load(meta.parent / (field + ".npy"), allow_pickle=False)
                save_image(map_rgb(values, ERROR_CMAP, norms[field]), dest / (field + ".png"), args.zoom)
            rendered += 1
        (output / dataset / "display_settings.json").write_text(
            json.dumps(notes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rendered {rendered} result folders under {output}")
    return 0 if rendered else 1


if __name__ == "__main__":
    raise SystemExit(main())
