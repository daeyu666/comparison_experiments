from pathlib import Path

import numpy as np
import pytest
import torch

import test_real as batch
from comparison.real_augsburg.common import pad_pair, positions
from experiment_protocol import (
    REAL_CHECKPOINT_FAMILY, REAL_CLASSICAL_METHODS, REAL_METHODS,
    build_real_native_test_command, build_real_train_command, resolve_model,
)


def weight(root, method, family, name):
    p = root / "comparison" / method / "checkpoints" / family / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()
    return p


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
