import csv
import json
import subprocess
import sys

import numpy as np
from PIL import Image
import pytest

import visualize_real_augsburg as batch


def fixture(tmp_path):
    cache = tmp_path / "cache"
    (cache / "full").mkdir(parents=True)
    roi = dict(protocol_id="Augsburg2-Wald-center-holdout-v1", source_region="sub_area_2",
               test_bbox_30m=[24, 36, 72, 84], test_bbox_10m=[72, 108, 216, 252])
    (cache / "roi.json").write_text(json.dumps(roi))
    msi = np.random.default_rng(12).random((300, 360, 4), dtype=np.float32)
    np.save(cache / "full/hr_msi.npy", msi)
    return cache


def test_archive_variant_selection_never_falls_back(tmp_path):
    root = tmp_path / "results"
    for folder in (root / "EMR-Diff/base128", root / "EMR-Diff/default", root / "S2Diff-MH/default"):
        folder.mkdir(parents=True)
        (folder / "fused_hsi.npy").touch()
    assert batch.find_cube(root, "EMR-Diff")[0].parent.name == "base128"
    assert batch.find_cube(root, "ours")[0].parent.parent.name == "S2Diff-MH"
    (root / "EMR-Diff/base128/fused_hsi.npy").unlink()
    with pytest.raises(FileNotFoundError):
        batch.find_cube(root, "EMR-Diff")


def test_all_twelve_share_stretch_and_overview_is_not_rewritten(tmp_path):
    cache = fixture(tmp_path)
    root = tmp_path / "archive"
    output = tmp_path / "figures"
    template = np.random.default_rng(3).random((144, 144, 242), dtype=np.float32).astype(np.float16)
    paths = []
    for index, method in enumerate(batch.METHODS):
        folder = root / ("S2Diff-MH" if method == "ours" else method) / ("base128" if method == "EMR-Diff" else "default")
        folder.mkdir(parents=True)
        cube = template * (0.5 + index / 20)
        path = folder / "fused_hsi.npy"
        np.save(path, cube)
        paths.append(path)
    original_times = {path: path.stat().st_mtime_ns for path in paths}
    argv = ["--results_root", str(root), "--wald_root", str(cache), "--output_dir", str(output)]
    assert batch.main(argv) == 0
    settings = json.loads((output / "visualization_provenance.json").read_text())
    assert len(settings["results"]) == 12
    assert all(row["status"] == "ok" for row in settings["results"])
    with (output / "visualization_summary.csv").open(encoding="utf-8-sig", newline="") as f:
        assert len(list(csv.DictReader(f))) == 12
    assert settings["inference_triggered_by_visualization"] is False
    assert settings["hsi_rgb_indices_0based"] == [43, 28, 10]
    legacy = batch.original_renderer()
    bounds = np.array(settings["shared_bounds"])
    reference = legacy.stretch(np.load(paths[0])[..., [43, 28, 10]], bounds)
    actual = Image.open(output / "CNMF/default/reconstruction_rgb.png")
    np.testing.assert_array_equal(np.asarray(actual), np.asarray(reference))
    overview = output / "Augsburg2_original_MSI_redbox.png"
    before = overview.stat().st_mtime_ns
    assert Image.open(overview).getpixel((108, 72)) == (255, 0, 0)
    assert len(list(output.rglob("*redbox.png"))) == 1
    assert batch.main(argv) == 0
    assert overview.stat().st_mtime_ns == before
    assert original_times == {path: path.stat().st_mtime_ns for path in paths}
    subprocess.run([
        sys.executable, str(batch.ROOT / "comparison/EMR-Diff/visualize_augsburg2_wald_center_holdout.py"),
        "--all_saved", "--results_root", str(root), "--wald_root", str(cache), "--output_dir", str(output),
    ], check=True)
    assert overview.stat().st_mtime_ns == before


def test_wrong_full_scene_result_is_rejected(tmp_path):
    file = tmp_path / "full.npy"
    np.save(file, np.zeros((300, 360, 242), dtype=np.float16))
    with pytest.raises(ValueError, match="Expected heldout"):
        batch.reconstruction_rgb(file, (144, 144, 242), (43, 28, 10))
