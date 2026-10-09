"""Visualize saved EMR-Diff Augsburg-2 center-heldout reconstruction.

This script is read-only with respect to models: it NEVER trains or runs
inference. It only loads already-saved 144x144x242 heldout HSI arrays.

Default: EMR-Diff only.
Use --compare_all to render S2Diff-MH + UAFL + EMR-Diff together with the
same HSI RGB bands and shared percentile stretch.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

THIS = Path(__file__).resolve().parent
UAFL_ROOT = THIS.parent / "UAFL"
if str(UAFL_ROOT) not in sys.path:
    sys.path.insert(0, str(UAFL_ROOT))

from augsburg2_wald_center_roi import read_roi


DEFAULT_EMR = (
    "./comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout/"
    "Augsburg2_Wald_EMRDiff_heldout_HSI.npy"
)
DEFAULT_UAFL = (
    "./comparison/UAFL/outputs/augsburg2_wald_center_holdout/"
    "Augsburg2_Wald_UAFL_heldout_HSI.npy"
)
DEFAULT_S2DIFF = (
    "../S2Diff-MH/outputs/augsburg2_wald_center_holdout/"
    "Augsburg2_Wald_heldout_HSI.npy"
)


def stretch(rgb, bounds=None):
    x = np.asarray(rgb, dtype=np.float32)
    if bounds is None:
        lo = np.percentile(x.reshape(-1, 3), 1, axis=0)
        hi = np.percentile(x.reshape(-1, 3), 99, axis=0)
    else:
        lo, hi = bounds
    scaled = (x - lo[None, None]) / np.maximum(
        hi - lo, 1e-6
    )[None, None]
    return Image.fromarray(
        np.asarray(np.rint(np.clip(scaled, 0, 1) * 255), dtype=np.uint8),
        "RGB",
    )


def resolve_methods(methods, wald_root):
    roi = read_roi(wald_root)
    if roi is None:
        raise FileNotFoundError(
            f"Center-holdout roi.json missing: {Path(wald_root) / 'roi.json'}"
        )
    y0, x0, y1, x1 = map(int, roi["test_bbox_10m"])
    expected = (y1 - y0, x1 - x0, 242)
    resolved = []
    missing = []

    for name, file_path in methods:
        model_name = name.strip()
        if not model_name:
            raise ValueError("--method requires a non-empty model name")
        path = Path(file_path).expanduser().resolve()
        if not path.is_file():
            missing.append((model_name, path))
            continue
        arr = np.load(path, mmap_mode="r")
        if arr.shape != expected:
            raise ValueError(
                f"{model_name}: reconstruction shape={arr.shape}, "
                f"expected heldout {expected}: {path}. "
                "Do not use old full-region outputs."
            )
        resolved.append((model_name, path))

    if missing:
        lines = [f"{name}: {path}" for name, path in missing]
        commands = [
            "EMR-Diff: python comparison/EMR-Diff/infer_augsburg2_wald.py --center_holdout --write_tif",
            "UAFL: python comparison/UAFL/infer_augsburg2_wald.py --center_holdout --write_tif",
            "S2Diff-MH: python infer_augsburg2_wald.py --center_holdout --write_tif",
        ]
        raise FileNotFoundError(
            "Visualization never performs inference. Missing saved heldout HSI:\n"
            + "\n".join(lines)
            + "\nRun the corresponding inference command first:\n"
            + "\n".join(commands)
        )
    return roi, resolved


def visualize(wald_root, output_dir, methods, rgb, savefig):
    roi, methods = resolve_methods(methods, wald_root)
    root = Path(wald_root)

    if roi.get("source_region") != "sub_area_2":
        raise ValueError("Expected Augsburg Region-2 center holdout")

    y0, x0, y1, x1 = map(int, roi["test_bbox_10m"])
    full_msi = np.load(root / "full" / "hr_msi.npy", mmap_mode="r")
    if full_msi.ndim != 3 or full_msi.shape[-1] != 4:
        raise ValueError("Expected original real four-band Sentinel-2 MSI")

    # Visualization only: S2 RGB uses B4/B3/B2; no fitted radiometry is used.
    full_rgb = np.asarray(full_msi[..., [2, 1, 0]], dtype=np.float32)
    overview = stretch(full_rgb)
    draw = ImageDraw.Draw(overview)
    draw.rectangle((x0, y0, x1 - 1, y1 - 1), outline=(255, 0, 0), width=3)
    raw_roi = stretch(full_rgb[y0:y1, x0:x1])

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overview_path = out / "Augsburg2_original_MSI_redbox.png"
    roi_path = out / "Augsburg2_original_MSI_heldout_ROI.png"
    overview.save(overview_path)
    raw_roi.save(roi_path)

    model_rgbs = {}
    for name, path in methods:
        cube = np.load(path, mmap_mode="r")
        model_rgbs[name] = np.asarray(
            cube[:, :, list(rgb)], dtype=np.float32
        )

    # One shared stretch across all reconstructed HSI methods. This avoids
    # per-method contrast tuning that could make visual comparison misleading.
    values = np.concatenate(
        [arr.reshape(-1, 3) for arr in model_rgbs.values()], axis=0
    )
    shared_bounds = (
        np.percentile(values, 1, axis=0),
        np.percentile(values, 99, axis=0),
    )

    model_rgb_files = {}
    panels = [
        ("Original S2 MSI (red box)", overview),
        ("Observed 10m MSI ROI", raw_roi),
    ]
    for name, path in methods:
        image = stretch(model_rgbs[name], shared_bounds)
        source = Path(path)
        rgb_path = source.with_name(
            source.stem.replace("_HSI", "") + "_RGB.png"
        )
        image.save(rgb_path)
        model_rgb_files[name] = str(rgb_path.resolve())
        panels.append((f"{name} (held-out reconstruction)", image))

    tile_w, tile_h, title_h = 290, 290, 37
    canvas = Image.new(
        "RGB", (tile_w * len(panels), tile_h + title_h), "#10151e"
    )
    painter = ImageDraw.Draw(canvas)
    for index, (label, panel) in enumerate(panels):
        image = panel.copy()
        image.thumbnail(
            (tile_w - 10, tile_h - 10), Image.Resampling.LANCZOS
        )
        px = (
            index * tile_w
            + 5
            + (tile_w - 10 - image.width) // 2
        )
        py = title_h + (tile_h - image.height) // 2
        canvas.paste(image, (px, py))
        painter.text((index * tile_w + 8, 11), label, fill="white")

    figure = Path(savefig).expanduser()
    if figure.suffix.lower() != ".png":
        raise ValueError("--savefig requires a .png filename")
    figure.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(figure)

    provenance = {
        "protocol_id": roi["protocol_id"],
        "source_region": roi["source_region"],
        "bbox_10m": roi["test_bbox_10m"],
        "bbox_30m": roi["test_bbox_30m"],
        "rgb_indices_0based_HSI": list(rgb),
        "reconstruction_shared_percentiles": [1, 99],
        "model_paths": {
            name: str(path.resolve()) for name, path in methods
        },
        "model_rgb_files": model_rgb_files,
        "overview": str(overview_path.resolve()),
        "observed_msi_roi": str(roi_path.resolve()),
        "figure": str(figure.resolve()),
        "inference_triggered_by_visualization": False,
        "metrics_unchanged_by_visualization": True,
    }
    provenance_path = out / "visualization_provenance.json"
    provenance_path.write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"VISUALIZATION={figure.resolve()}")
    for name, file_path in model_rgb_files.items():
        print(f"{name}_RGB={file_path}")
    print(f"REDBOX_BBOX_10M={roi['test_bbox_10m']}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Visualize existing EMR/S2Diff/UAFL Augsburg center-heldout "
            "reconstructions; NEVER trains or infers"
        )
    )
    parser.add_argument(
        "--wald_root",
        default="../S2Diff-MH/data/augsburg2_wald_center_holdout",
    )
    parser.add_argument(
        "--output_dir",
        default=(
            "./comparison/EMR-Diff/outputs/"
            "augsburg2_wald_center_holdout/visualization"
        ),
    )
    parser.add_argument(
        "--method",
        nargs=2,
        action="append",
        default=[],
        metavar=("NAME", "FUSED_NPY"),
        help=(
            "Existing 144x144x242 heldout reconstruction; repeat for "
            "multiple models"
        ),
    )
    parser.add_argument(
        "--compare_all",
        action="store_true",
        help="Compare S2Diff-MH, UAFL and EMR-Diff saved outputs",
    )
    parser.add_argument("--hsi_rgb", default="43,28,10")
    parser.add_argument(
        "--savefig",
        default=(
            "./comparison/EMR-Diff/outputs/"
            "augsburg2_wald_center_holdout/"
            "Augsburg_holdout_EMRDiff_RGB.png"
        ),
    )
    args = parser.parse_args()

    bands = tuple(int(value) for value in args.hsi_rgb.split(","))
    if len(bands) != 3 or any(value < 0 or value >= 242 for value in bands):
        raise ValueError(
            "--hsi_rgb requires three zero-based HSI band indices in [0,241]"
        )

    if args.method:
        methods = args.method
    elif args.compare_all:
        methods = [
            ("S2Diff", DEFAULT_S2DIFF),
            ("UAFL", DEFAULT_UAFL),
            ("EMR-Diff", DEFAULT_EMR),
        ]
    else:
        methods = [("EMR-Diff", DEFAULT_EMR)]

    visualize(
        args.wald_root,
        args.output_dir,
        methods,
        rgb=bands,
        savefig=args.savefig,
    )


if __name__ == "__main__":
    main()
