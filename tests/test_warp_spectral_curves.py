import csv
import json
from types import SimpleNamespace

import numpy as np
import pytest

import plot_warp_spectral_curves as curves


def args(**overrides):
    return SimpleNamespace(paviau_pixel=None, cave_pixel=None, cave_sample=None,
                           radius=1, margin=8, dark_floor=.01, clip_predictions=False, **overrides)


def archive(root):
    for dataset, bands in (("PaviaU", 103), ("CAVE", 31)):
        sample = "center" if dataset == "PaviaU" else "jelly_beans__r192_c192"
        gt = np.broadcast_to(np.linspace(.1, .7, bands)[:, None, None], (bands, 32, 32)).astype(np.float32).copy()
        gt[:, :, 16:] = np.linspace(.7, .1, bands)[:, None, None]
        hashes = dict(gt=curves.tensor_hash(gt), lr_warp="one-common-warp")
        meta = dict(dataset=dataset, sample=sample, scene="center" if dataset == "PaviaU" else "jelly_beans",
                    crop_rc_hw=[192,192,32,32], tensor_sha256=hashes,
                    wavelengths_nm=np.linspace(430 if dataset == "PaviaU" else 400, 860 if dataset == "PaviaU" else 700, bands).tolist())
        relative = f"raw/{dataset}/inputs/{sample}.npz"
        (root / relative).parent.mkdir(parents=True)
        np.savez(root / relative, gt=gt, metadata=json.dumps(meta))
        for index, method in enumerate(curves.METHODS):
            dest = root / "raw" / dataset / "unregistered/warp" / method / sample
            dest.mkdir(parents=True)
            np.save(dest / "prediction.npy", gt * (1 + index / 100))
            row = dict(identity=dict(dataset=dataset, group="unregistered", condition="warp", method=method,
                                    sample=sample, checkpoint={}, seed=1234, input_sha256=hashes), input_file=relative)
            (dest / "metadata.json").write_text(json.dumps(row))


def test_heterogeneity_measures_spectral_shape_not_brightness():
    signature = np.linspace(.2, .8, 31)
    brightness = np.linspace(.1, 1, 32)[None, :].repeat(32, axis=0)
    same_shape = signature[:,None,None] * brightness[None]
    score, eligible, _ = curves.spectral_heterogeneity(same_shape)
    assert np.max(score[eligible]) < 1e-12
    changed = same_shape.copy()
    changed[:, :, 16:] = signature[::-1,None,None] * brightness[None,:,16:]
    score, eligible, _ = curves.spectral_heterogeneity(changed)
    row, col = np.unravel_index(np.argmax(np.where(eligible,score,-np.inf)),score.shape)
    assert col in (15, 16) and score[row,col] > .01


def test_selection_uses_gt_and_rejects_mixed_inputs(tmp_path):
    archive(tmp_path)
    first = curves.choose_pixel(tmp_path, "PaviaU", args())
    pred = tmp_path / "raw/PaviaU/unregistered/warp/ours/center/prediction.npy"
    np.save(pred, np.zeros_like(first["gt"]))
    second = curves.choose_pixel(tmp_path, "PaviaU", args())
    assert first["selected"]["pixel_rc_0based"] == second["selected"]["pixel_rc_0based"]
    meta_path = pred.parent / "metadata.json"
    meta = json.loads(meta_path.read_text())
    meta["identity"]["input_sha256"]["lr_warp"] = "different-warp"
    meta_path.write_text(json.dumps(meta))
    with pytest.raises(ValueError, match="different observations"):
        curves.choose_pixel(tmp_path, "PaviaU", args())


def test_complete_warp_figures_and_csv(tmp_path):
    archive(tmp_path)
    sources = {p: p.stat().st_mtime_ns for p in (tmp_path / "raw").rglob("*") if p.is_file()}
    assert curves.main(["--input_dir",str(tmp_path)]) == 0
    out = tmp_path / "spectral_curves"
    selected = json.loads((out / "selected_pixels.json").read_text())
    assert set(selected) == {"PaviaU", "CAVE"}
    assert all(x["condition"] == "warp" for x in selected.values())
    for dataset, bands in (("PaviaU",103),("CAVE",31)):
        with (out / dataset / "spectra.csv").open(encoding="utf-8-sig",newline="") as handle:
            rows=list(csv.DictReader(handle))
        assert len(rows) == bands
        assert tuple(rows[0]) == ("wavelength_nm", "GT", *curves.METHODS)
        for ext in ("png","pdf","svg"):
            assert (out / dataset / f"spectral_curves.{ext}").is_file()
            assert (out / f"PaviaU_CAVE_warp_spectral_curves.{ext}").is_file()
    assert all(p.stat().st_mtime_ns==time for p,time in sources.items())
    missing=tmp_path / "raw/CAVE/unregistered/warp/UAFL/jelly_beans__r192_c192/prediction.npy"
    missing.unlink()
    with pytest.raises(FileNotFoundError,match="all six methods"):
        curves.complete_samples(tmp_path,"CAVE")
