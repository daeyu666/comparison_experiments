"""Plot bandwise absolute errors at a GT-selected pixel from saved Warp cubes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATASETS = ("PaviaU", "CAVE")
METHODS = ("M2DTN", "PMI-RFCoNet", "MoE-PNP", "SA-RFM", "UAFL", "ours")
STYLES = {
    "GT": dict(color="black", linestyle="-", linewidth=2.2, zorder=10),
    "M2DTN": dict(color="#0072B2", linestyle="-", linewidth=1.3),
    "PMI-RFCoNet": dict(color="#E69F00", linestyle="-", linewidth=1.5),
    "MoE-PNP": dict(color="#009E73", linestyle="-", linewidth=1.3),
    "SA-RFM": dict(color="#CC79A7", linestyle="-", linewidth=1.3),
    "UAFL": dict(color="#56B4E9", linestyle="-", linewidth=1.3),
    "ours": dict(color="#D55E00", linestyle="-", linewidth=2.0, zorder=9),
}


def tensor_hash(array):
    value = np.ascontiguousarray(array, dtype=np.float32)
    return hashlib.sha256(str(value.shape).encode() + value.tobytes()).hexdigest()


def spectral_heterogeneity(gt, radius=1, margin=8, dark_floor=0.01):
    """Local mean pairwise cosine dissimilarity of unit-norm GT spectra."""
    if gt.ndim != 3 or not np.isfinite(gt).all():
        raise ValueError("GT must be finite CHW")
    c, h, w = gt.shape
    if min(h, w) <= 2 * max(margin, radius):
        raise ValueError("The selected margin/window leaves no interior pixels")
    x = np.asarray(gt, dtype=np.float64)
    norms = np.linalg.norm(x, axis=0)
    valid = norms > 1e-8
    unit = np.divide(x, norms[None], out=np.zeros_like(x), where=valid[None])
    padded = np.pad(unit, ((0, 0), (radius, radius), (radius, radius)), mode="reflect")
    valid_pad = np.pad(valid, radius, mode="reflect")
    total = np.zeros((c, h, w))
    count = np.zeros((h, w))
    for row in range(2 * radius + 1):
        for col in range(2 * radius + 1):
            total += padded[:, row:row+h, col:col+w]
            count += valid_pad[row:row+h, col:col+w]
    mean_unit = total / count.clip(1)[None]
    # n/(n-1) converts variance about the mean to mean pairwise cosine distance.
    scores = (count / (count - 1).clip(1)) * (1 - (mean_unit**2).sum(axis=0))
    scores = np.maximum(scores, 0)
    brightness = x.mean(axis=0)
    positive = brightness[valid & (brightness > dark_floor)]
    threshold = max(dark_floor, float(np.percentile(positive, 20))) if positive.size else dark_floor
    eligible = (brightness >= threshold) & valid & (count == (2 * radius + 1)**2)
    border = max(margin, radius)
    eligible[:border] = eligible[-border:] = False
    eligible[:, :border] = eligible[:, -border:] = False
    return scores, eligible, threshold


def complete_samples(root, dataset):
    base = root / "raw" / dataset / "unregistered/warp"
    available = {method: {p.name for p in (base / method).glob("*") if p.is_dir()
                         and (p / "prediction.npy").is_file() and (p / "metadata.json").is_file()}
                 for method in METHODS}
    common = set.intersection(*(available[m] for m in METHODS))
    if not common:
        counts = ", ".join(f"{m}={len(available[m])}" for m in METHODS)
        raise FileNotFoundError(f"{dataset}: no Warp sample shared by all six methods under {base}; {counts}. "
                                "Run export_paper_predictions.py first; this script never infers missing results.")
    return base, sorted(common)


def read_sample(root, base, dataset, sample):
    records = {}
    for method in METHODS:
        path = base / method / sample / "metadata.json"
        row = json.loads(path.read_text(encoding="utf-8"))
        identity = row["identity"]
        expected = dict(dataset=dataset, group="unregistered", condition="warp", method=method, sample=sample)
        if any(identity.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Result is not the requested Warp sample: {path}")
        records[method] = row
    first = records[METHODS[0]]
    input_path = (root / first["input_file"]).resolve()
    with np.load(input_path, allow_pickle=False) as saved:
        gt = saved["gt"]
        meta = json.loads(saved["metadata"].item())
    if meta.get("dataset") != dataset or meta.get("sample") != sample:
        raise ValueError(f"Saved GT dataset/sample mismatch: {input_path}")
    if tensor_hash(gt) != meta["tensor_sha256"]["gt"]:
        raise ValueError(f"GT data no longer matches its export hash: {input_path}")
    for method, row in records.items():
        if row["identity"]["input_sha256"] != meta["tensor_sha256"]:
            raise ValueError(f"{dataset}/{method}/{sample} uses different observations")
        if (root / row["input_file"]).resolve() != input_path:
            raise ValueError(f"{dataset}/{method}/{sample} references another input file")
    wavelengths = np.asarray(meta["wavelengths_nm"], dtype=float)
    if (wavelengths.shape != (gt.shape[0],) or not np.isfinite(wavelengths).all()
            or np.any(np.diff(wavelengths) <= 0)):
        raise ValueError("Expected finite increasing HSI wavelengths from the saved input")
    return gt, meta, records, wavelengths


def choose_pixel(root, dataset, args):
    base, samples = complete_samples(root, dataset)
    manual = args.paviau_pixel if dataset == "PaviaU" else args.cave_pixel
    frozen = getattr(args, "saved_selection", {}).get(dataset)
    if manual is not None or (dataset == "CAVE" and args.cave_sample):
        frozen = None
    if frozen:
        if frozen.get("dataset") != dataset or frozen.get("condition") != "warp":
            raise ValueError(f"Invalid saved Warp selection for {dataset}")
        if frozen["sample"] not in samples:
            raise ValueError(f"Previously selected {dataset}/{frozen['sample']} is incomplete; use --reselect")
        samples = [frozen["sample"]]
        manual = frozen["pixel_rc_0based"]
    candidates, checked = [], []
    for sample in samples:
        gt, meta, records, wavelengths = read_sample(root, base, dataset, sample)
        if dataset == "CAVE" and args.cave_sample and args.cave_sample not in (sample, meta["scene"]):
            continue
        scores, eligible, threshold = spectral_heterogeneity(gt, args.radius, args.margin, args.dark_floor)
        if not eligible.any():
            continue
        ranked = np.where(eligible, scores, -np.inf)
        r, c = np.unravel_index(np.argmax(ranked), ranked.shape)
        checked.append(dict(sample=sample, scene=meta["scene"], max_heterogeneity=float(scores[r, c]),
                            eligible_pixels=int(eligible.sum()), brightness_threshold=threshold))
        candidates.append((float(scores[r, c]), sample, int(r), int(c)))
    if not candidates:
        raise ValueError(f"{dataset}: no complete eligible sample matches the selection")
    if manual is not None:
        if len(candidates) != 1:
            raise ValueError("Manual pixel needs one exact sample; use --cave_sample with the full crop ID")
        _, sample, _, _ = candidates[0]
        r, c = manual
    else:
        # Sorted traversal and strict greater-than keep tie-breaking deterministic.
        _, sample, r, c = max(candidates, key=lambda item: item[0])
    gt, meta, records, wavelengths = read_sample(root, base, dataset, sample)
    if frozen and frozen["input_sha256"] != meta["tensor_sha256"]:
        raise ValueError(f"Previously selected {dataset} observations changed; use --reselect or another output directory")
    scores, eligible, threshold = spectral_heterogeneity(gt, args.radius, args.margin, args.dark_floor)
    if not (0 <= r < gt.shape[1] and 0 <= c < gt.shape[2]) or not eligible[r, c]:
        raise ValueError(f"{dataset}: pixel {(r,c)} is dark, on the excluded border, or outside the image")
    if manual is None and scores[r, c] <= 1e-8:
        raise ValueError(f"{dataset}: GT has no detectable local spectral heterogeneity")
    spectra, sources = {"GT": gt[:, r, c].astype(float)}, {}
    for method in METHODS:
        path = base / method / sample / "prediction.npy"
        cube = np.load(path, mmap_mode="r", allow_pickle=False)
        if cube.shape != gt.shape:
            raise ValueError(f"Wrong prediction shape: {path}")
        vector = np.asarray(cube[:, r, c], dtype=float)
        if not np.isfinite(vector).all():
            raise ValueError(f"Nonfinite spectrum: {path}")
        spectra[method] = vector.clip(0, 1) if args.clip_predictions else vector
        sources[method] = dict(prediction=str(path), checkpoint=records[method]["identity"]["checkpoint"],
                               exported_seed=records[method]["identity"].get("seed"))
    crop = meta.get("crop_rc_hw")
    selected = dict(dataset=dataset, condition="warp", sample=sample, scene=meta["scene"],
                    pixel_rc_0based=[r, c], crop_rc_hw=crop,
                    scene_pixel_rc_0based=None if crop is None else [int(crop[0])+r, int(crop[1])+c],
                    selection=(frozen["selection"] if frozen else "manual" if manual else
                               "maximum_GT_local_pairwise_cosine_dissimilarity"),
                    heterogeneity=float(scores[r,c]), radius=args.radius, margin=args.margin,
                    dark_floor=args.dark_floor, brightness_threshold=threshold,
                    input_sha256=meta["tensor_sha256"], warp=meta.get("warp"),
                    clip_predictions=args.clip_predictions,
                    candidate_samples=frozen.get("candidate_samples", checked) if frozen else checked, sources=sources)
    return dict(gt=gt, meta=meta, scores=scores, selected=selected, wavelengths=wavelengths, spectra=spectra)


def absolute_errors(spectra):
    return {name: np.abs(spectra[name] - spectra["GT"]) for name in METHODS}


def draw_curves(axis, data, plot_type="error", x_axis="wavelength"):
    values = absolute_errors(data["spectra"]) if plot_type == "error" else data["spectra"]
    x = data["wavelengths"] if x_axis == "wavelength" else np.arange(1, len(data["wavelengths"]) + 1)
    for name, spectrum in values.items():
        axis.plot(x, spectrum, label="Ours" if name == "ours" else name, **STYLES[name])
    r, c = data["selected"]["pixel_rc_0based"]
    axis.set_title(f"{data['selected']['dataset']} (Warp), pixel ({r}, {c})", fontsize=11)
    axis.set_xlabel("Wavelength (nm)" if x_axis == "wavelength" else "Band")
    axis.set_ylabel("Absolute spectral error" if plot_type == "error" else "Normalized spectral intensity")
    axis.set_xlim(x[[0, -1]])
    if plot_type == "error":
        axis.set_ylim(0, max(1e-6, max(float(v.max()) for v in values.values())) * 1.08)
    axis.grid(True, alpha=.2, linewidth=.5)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure, stem):
    for extension in ("png", "pdf", "svg"):
        figure.savefig(stem.with_suffix("." + extension), dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def save_diagnostics(data, out):
    gt, wavelengths = data["gt"], data["wavelengths"]
    bands = [int(np.argmin(abs(wavelengths - nm))) for nm in (650, 550, 470)]
    rgb = gt[bands].clip(0, 1)
    lo, hi = np.percentile(rgb, [1, 99], axis=(1, 2))
    rgb = ((rgb - lo[:, None, None]) / (hi-lo).clip(1e-6)[:, None, None]).clip(0, 1)
    image = Image.fromarray(np.round(rgb.transpose(1,2,0)*255).astype(np.uint8))
    zoom = 4
    image = image.resize((image.width*zoom, image.height*zoom), Image.Resampling.NEAREST)
    r, c = data["selected"]["pixel_rc_0based"]
    x, y = c*zoom+zoom//2, r*zoom+zoom//2
    painter = ImageDraw.Draw(image)
    painter.ellipse((x-9, y-9, x+9, y+9), outline="white", width=4)
    painter.ellipse((x-8, y-8, x+8, y+8), outline="red", width=2)
    painter.line((x-4,y,x+4,y), fill="red", width=2)
    painter.line((x,y-4,x,y+4), fill="red", width=2)
    image.save(out / "selected_pixel_rgb.png")
    np.save(out / "gt_heterogeneity.npy", data["scores"].astype(np.float32), allow_pickle=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input_dir", type=Path, default=ROOT / "outputs/paper_visualization")
    p.add_argument("--output_dir", type=Path, help="Default: INPUT_DIR/spectral_curves")
    p.add_argument("--paviau_pixel", nargs=2, type=int, metavar=("ROW", "COL"))
    p.add_argument("--cave_pixel", nargs=2, type=int, metavar=("ROW", "COL"))
    p.add_argument("--cave_sample", help="CAVE scene name or exact crop ID; default searches all complete scenes")
    p.add_argument("--radius", type=int, default=1, help="GT heterogeneity window radius (default: 3x3 window)")
    p.add_argument("--margin", type=int, default=8)
    p.add_argument("--dark_floor", type=float, default=.01)
    p.add_argument("--clip_predictions", action="store_true", help="Explicitly plot predictions clipped to [0,1]")
    p.add_argument("--plot_type", choices=("error", "spectrum"), default="error",
                   help="Default: absolute error |prediction-GT|; spectrum retains the original intensity plot")
    p.add_argument("--x_axis", choices=("wavelength", "band"), default="wavelength")
    p.add_argument("--selection_json", type=Path, help="Reuse previously selected pixels (default: OUTPUT_DIR/selected_pixels.json if present)")
    p.add_argument("--reselect", action="store_true", help="Select pixels again from GT instead of reusing saved coordinates")
    args = p.parse_args(argv)
    if args.radius < 1 or args.margin < 0 or not np.isfinite(args.dark_floor) or not 0 <= args.dark_floor < 1:
        p.error("Require radius>=1, margin>=0, and 0<=dark_floor<1")
    root = args.input_dir.expanduser().resolve()
    output = (args.output_dir or root / "spectral_curves").expanduser().resolve()
    if args.reselect and args.selection_json:
        p.error("Use --reselect or --selection_json, not both")
    selection_path = (args.selection_json.expanduser().resolve() if args.selection_json
                      else output / "selected_pixels.json")
    args.saved_selection = (json.loads(selection_path.read_text(encoding="utf-8"))
                            if not args.reselect and selection_path.is_file() else {})
    if args.selection_json and not selection_path.is_file():
        raise FileNotFoundError(selection_path)
    data = [choose_pixel(root, dataset, args) for dataset in DATASETS]
    output.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10,
                         "pdf.fonttype": 42, "svg.fonttype": "none"})
    selections = {}
    for item in data:
        dataset = item["selected"]["dataset"]
        out = output / dataset
        out.mkdir(parents=True, exist_ok=True)
        with (out / "spectra.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            names = tuple(item["spectra"])
            writer.writerow(["wavelength_nm", *names])
            writer.writerows(zip(item["wavelengths"], *(item["spectra"][name] for name in names)))
        errors = absolute_errors(item["spectra"])
        with (out / "absolute_errors.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.writer(handle)
            writer.writerow(["band_1based", "wavelength_nm", *METHODS])
            writer.writerows(zip(range(1, len(item["wavelengths"])+1), item["wavelengths"], *errors.values()))
        save_diagnostics(item, out)
        fig, axis = plt.subplots(figsize=(6.2, 4.0))
        draw_curves(axis, item, args.plot_type, args.x_axis)
        axis.legend(ncol=3 if args.plot_type == "error" else 4, loc="lower center", bbox_to_anchor=(.5,1.12), fontsize=8, frameon=False)
        fig.subplots_adjust(top=.73)
        stem = "spectral_absolute_error" if args.plot_type == "error" else "spectral_curves"
        save_figure(fig, out / stem)
        item["selected"].update(plot_type=args.plot_type, x_axis=args.x_axis)
        selections[dataset] = item["selected"]
        print(f"SELECTED {dataset}: {item['selected']['sample']} pixel={item['selected']['pixel_rc_0based']} "
              f"heterogeneity={item['selected']['heterogeneity']:.6g}", flush=True)
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 4.2))
    for axis, item in zip(axes, data):
        draw_curves(axis, item, args.plot_type, args.x_axis)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False, fontsize=9)
    figure.subplots_adjust(top=.80, bottom=.16, left=.07, right=.99, wspace=.25)
    save_figure(figure, output / f"PaviaU_CAVE_warp_{stem}")
    (output / "selected_pixels.json").write_text(json.dumps(selections, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"SPECTRAL_CURVES {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
