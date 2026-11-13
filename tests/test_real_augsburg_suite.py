import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import experiment_protocol as protocol
import test_real as batch
from comparison.real_augsburg.common import WaldCache, pad_pair, positions
from experiment_protocol import (
    REAL_CHECKPOINT_FAMILY, REAL_CLASSICAL_METHODS, REAL_METHODS,
    build_real_native_test_command, build_real_train_command, resolve_model,
)


def weight(root, method, family, name):
    p = root / "comparison" / method / "checkpoints" / family / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


@pytest.mark.parametrize("layout", ["nested", "direct", "legacy"])
def test_real_launchers_find_wald_data(tmp_path, monkeypatch, layout):
    repo = tmp_path / "comparison_experiments"
    monkeypatch.setattr(protocol, "ROOT", repo)
    name = protocol.REAL_WALD_CACHE_NAME
    legacy_data = tmp_path / "S2Diff-MH" / "data"
    legacy_cache = legacy_data / name
    legacy_meta = legacy_cache / "train" / "meta.json"
    legacy_meta.parent.mkdir(parents=True)
    legacy_meta.touch()
    data = legacy_data if layout == "legacy" else repo / "data"
    cache = data if layout == "direct" else data / name
    if layout != "legacy":
        meta = cache / "train" / "meta.json"
        meta.parent.mkdir(parents=True)
        meta.touch()
    calibration = data / "calibration" / protocol.REAL_RADIOMETRY_NAME
    calibration.parent.mkdir()
    calibration.touch()

    train = build_real_train_command(resolve_model("ESFS"), "cpu")
    assert train[train.index("--wald_root") + 1] == str(cache)
    assert train[train.index("--radiometry_json") + 1] == str(calibration)
    test = batch.parse_args(["--model", "CNMF", "--dry_run"])
    assert (test.wald_root, test.radiometry_json) == (cache, calibration)
    custom_cache, custom_calibration = tmp_path / "custom", tmp_path / "custom.json"
    test = batch.parse_args(["--model", "CNMF", "--dry_run",
                             "--wald_root", str(custom_cache),
                             "--radiometry_json", str(custom_calibration)])
    assert (test.wald_root, test.radiometry_json) == (custom_cache, custom_calibration)


def test_discovery_matches_real_family_and_selects_best_per_width(tmp_path):
    other = tmp_path / "second_disk"
    w64 = weight(tmp_path, "EMR-Diff", REAL_CHECKPOINT_FAMILY, "best.pth.tar")
    w128 = weight(other, "EMR-Diff", REAL_CHECKPOINT_FAMILY, "best_base128.pth.tar")
    w192 = weight(other, "EMR-Diff", REAL_CHECKPOINT_FAMILY, "last_base192.pth.tar")
    weight(other, "EMR-Diff", REAL_CHECKPOINT_FAMILY, "last_base128.pth.tar")
    weight(other, "EMR-Diff", "physical/PaviaU", "best_base96.pth.tar")
    roots = (tmp_path, other)
    assert [j.checkpoint for j in batch.discover("EMR-Diff", roots)] == [w64, w128]
    assert [j.checkpoint for j in batch.discover("EMR-Diff", roots, width=128)] == [w128]
    assert [j.checkpoint for j in batch.discover("EMR-Diff", roots, allow_last=True)] == [w64, w128, w192]


def test_batch_resumes_from_csv_and_saves_progress(tmp_path, monkeypatch, capsys):
    output = tmp_path / "outputs"
    output.mkdir()
    (output / "summary.csv").write_text(
        "method,variant,status,backbone_width,QNR,Dlambda,Ds\n"
        "CNMF,solver,ok,,0.91,0.02,0.07\n"
        "EMR-Diff,base64,ok,64,0.82,0.10,0.09\n"
        "HySure,solver,failed,,,,\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(batch, "ROOT", tmp_path)
    for name in ("best.pth.tar", "best_base128.pth.tar"):
        weight(tmp_path, "EMR-Diff", REAL_CHECKPOINT_FAMILY, name)
    loads, calls = [], []
    monkeypatch.setattr(WaldCache, "load", lambda *args: loads.append(args))

    def results():
        with (output / "summary.csv").open(newline="", encoding="utf-8") as handle:
            return {(r["method"], r["variant"]): r for r in csv.DictReader(handle)}

    def fake_run(job, args, dest):
        calls.append((job.method, job.variant))
        if job.method == "HySure":
            assert results()["EMR-Diff", "base128"]["status"] == "ok"
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "protocol.json").write_text("{}", encoding="utf-8")
        return {"QNR": 0.7, "Dlambda": 0.1, "Ds": 0.2}

    monkeypatch.setattr(batch, "run_job", fake_run)
    options = ["--output_dir", str(output), "--device", "cpu"]
    assert batch.main([*options, "--model", "EMR-Diff", "HySure"]) == 0
    assert calls == [("EMR-Diff", "base128"), ("HySure", "solver")]
    assert results()["CNMF", "solver"]["QNR"] == "0.91"
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["results"][1]["backbone_width"] == 64

    assert batch.main([*options, "--model", "CNMF", "EMR-Diff", "HySure"]) == 0
    assert len(loads) == 1 and len(calls) == 2
    assert "SKIP_OK CNMF/solver" in capsys.readouterr().out

    assert batch.main([*options, "--model", "CNMF", "--rerun"]) == 0
    assert calls[-1] == ("CNMF", "solver")
    assert results()["CNMF", "solver"]["QNR"] == "0.7"


def test_explicit_synthetic_weights_are_not_accepted(tmp_path):
    p = weight(tmp_path, "EMR-Diff", "physical/PaviaU", "best_base128.pth.tar")
    args = batch.parse_args(["--model", "EMR-Diff", "--checkpoint", str(p), "--dry_run"])
    with pytest.raises(ValueError, match="Synthetic x4"):
        batch.plan(args)


@pytest.mark.parametrize("method", REAL_METHODS)
def test_all_methods_have_real_routes(method):
    adapter = resolve_model(method)
    cmd = build_real_native_test_command(adapter, "cuda:0", check_files=False)
    assert Path(cmd[1]).is_file()
    assert "--center_holdout" in cmd
    assert cmd[cmd.index("--qnr_window_hr") + 1] == "48"
    if method in REAL_CLASSICAL_METHODS:
        assert "--checkpoint" not in cmd
        assert cmd[cmd.index("--device") + 1] == "cpu"
        with pytest.raises(NotImplementedError, match="no training"):
            build_real_train_command(adapter, "cuda:0")
    else:
        assert "--checkpoint" in cmd
        cmd = build_real_train_command(adapter, "cuda:0")
        assert Path(cmd[1]).is_file()
        assert cmd[cmd.index("--train_patch_size") + 1] == "24"


def test_emr_width_routes_to_separate_real_checkpoints():
    adapter = resolve_model("EMR-Diff")
    cmd = build_real_train_command(adapter, "cpu", hidden_channels=128, resume="auto")
    assert cmd[cmd.index("--model_width") + 1] == "128"
    assert cmd[cmd.index("--resume") + 1].endswith("augsburg2_wald_center_holdout/last_base128.pth.tar")
    cmd = build_real_native_test_command(adapter, "cpu", hidden_channels=128, check_files=False)
    assert cmd[cmd.index("--checkpoint") + 1].endswith("augsburg2_wald_center_holdout/best_base128.pth.tar")


def test_native_grid_and_model_padding_preserve_observed_pixels():
    hsi = torch.rand(1, 2, 8, 8)
    msi = torch.rand(1, 3, 24, 24)
    lr, hr = pad_pair(hsi, msi, multiple=8, minimum=16)
    assert lr.shape[-2:] == (16, 16) and hr.shape[-2:] == (48, 48)
    torch.testing.assert_close(lr[..., :8, :8], hsi)
    torch.testing.assert_close(hr[..., :24, :24], msi)
    result = np.zeros(144)
    for start in positions(144, 96, 48):
        result[start:start + 96] += 1
    assert (result > 0).all()
