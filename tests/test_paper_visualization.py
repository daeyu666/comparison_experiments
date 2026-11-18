from types import SimpleNamespace

import numpy as np
import pytest

from data_loader import CAVE_TEST_SCENES
from export_paper_predictions import cave_samples, checkpoint_for, error_maps
from fixed_deformation_protocol import STRENGTHS, sample_fixed_geometry


def test_cave_center_is_separate_from_the_official_grid():
    grid = [(s, r, c, 128) for s in CAVE_TEST_SCENES
            for r in range(0, 512, 128) for c in range(0, 512, 128)]
    test = SimpleNamespace(samples=grid)
    assert len(cave_samples(test, "all-grid", None)) == 192
    assert cave_samples(test, "center", ["jelly_beans"]) == [("jelly_beans", 192, 192, 128)]
    assert cave_samples(test, "central-grid", ["jelly_beans"]) == [("jelly_beans", 128, 128, 128)]
    with pytest.raises(ValueError, match="Not held-out"):
        cave_samples(test, "center", ["balloons"])


def test_visualization_warp_reaches_the_selected_local_peak():
    _, meta = sample_fixed_geometry(128, 128, STRENGTHS[4], sample_index=1, case_index=0,
                                    exact_local_max=True)
    assert meta["local_max_hr_px"] == pytest.approx(4.0)
    assert meta["min_local_jacobian"] >= 0.5
    assert (meta["dx_hr_px"], meta["dy_hr_px"], meta["rotation_deg"]) == (4., 4., 2.)


def test_sam_masks_empty_spectra_and_preserves_angles():
    pred = np.array([[[1., 0., .35]], [[0., 0., .7]]])
    gt = np.array([[[0., 0., .5]], [[1., 0., 1.]]])
    rmse, sam = error_maps(pred, gt)
    np.testing.assert_allclose(rmse, [[1., 0., np.sqrt((.15**2 + .3**2) / 2)]])
    np.testing.assert_allclose(sam, [[90., np.nan, 0.]], equal_nan=True, atol=1e-5)


def test_emr_width_prefers_base128_and_keeps_training_stages_separate(tmp_path):
    reg = tmp_path / "EMR-Diff-checkpoints" / "physical" / "CAVE"
    reg.mkdir(parents=True)
    for name in ("best.pth.tar", "best_base128.pth.tar"):
        (reg / name).touch()
    assert checkpoint_for(tmp_path, "EMR-Diff", "physical", "CAVE", None).name == "best_base128.pth.tar"
    assert checkpoint_for(tmp_path, "EMR-Diff", "physical", "CAVE", 64).name == "best.pth.tar"
    with pytest.raises(FileNotFoundError):
        checkpoint_for(tmp_path, "EMR-Diff", "hsi_warp_final", "CAVE", None)
