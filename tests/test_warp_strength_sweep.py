from types import SimpleNamespace

import numpy as np
import torch

from test_warp_strength import STRENGTHS, paired_geometries, prepare_inputs, save_csv


def test_strengths_pair_geometry_and_preserve_training_bounds():
    from comparison.UAFL.hsi_deformation import jacobian_determinant
    first = paired_geometries(128, 128, 2, 3, 10)
    torch.manual_seed(999)
    replay = paired_geometries(128, 128, 2, 3, 10)
    for alpha, geometry, repeated in zip(STRENGTHS, first, replay):
        for key in ("dx", "dy", "theta_deg", "control", "local_field"):
            torch.testing.assert_close(getattr(geometry, key), getattr(first[-1], key) * alpha, atol=0, rtol=0)
            torch.testing.assert_close(getattr(geometry, key), getattr(repeated, key), atol=0, rtol=0)
        assert abs(float(geometry.dx)) <= 4 * alpha
        assert abs(float(geometry.dy)) <= 4 * alpha
        assert abs(float(geometry.theta_deg)) <= 2 * alpha
        assert float(torch.linalg.vector_norm(geometry.local_field, dim=1).max()) <= 4 * alpha + 1e-6
        assert float(jacobian_determinant(geometry.local_field).min()) >= .5


def test_common_inputs_preserve_exact_registered_observation(monkeypatch, tmp_path):
    import export_paper_predictions
    gt = torch.rand(3, 16, 16)
    lr = torch.rand(3, 4, 4)
    item = dict(gt=gt, hr_msi=gt[:2], lr_hsi=lr)

    class Dataset:
        degradation_operator = SimpleNamespace(degrade=lambda x: torch.nn.functional.avg_pool2d(x, 4))
        def __len__(self):
            return 1
        def __getitem__(self, index):
            assert index == 0
            return item

    monkeypatch.setattr(export_paper_predictions, "dataset_and_info", lambda cfg: (Dataset(), {"srf_weights": np.ones((2, 3)) / 3}))
    job = dict(dataset="PaviaU", data_root="unused", seed=10, cases=2)
    manifest = prepare_inputs(job, tmp_path / "one")
    again = prepare_inputs(job, tmp_path / "two")
    assert manifest["input_sha256"] == again["input_sha256"]
    with np.load(manifest["paths"][0]) as saved:
        np.testing.assert_array_equal(saved["observations"][0], np.stack([lr.numpy()] * 2))
        assert saved["observations"].shape == (5, 2, 3, 4, 4)
        assert not np.array_equal(saved["observations"][-1, 0], saved["observations"][-1, 1])


def test_csv_keeps_existing_unselected_rows(tmp_path):
    import csv
    path = tmp_path / "summary.csv"
    rows = {("CAVE", "ours", 1.): dict(dataset="CAVE", method="ours", strength=1., status="ok"),
            ("PaviaU", "UAFL", .25): dict(dataset="PaviaU", method="UAFL", strength=.25, status="failed", error="missing")}
    save_csv(path, rows)
    with path.open(encoding="utf-8-sig", newline="") as handle:
        saved = list(csv.DictReader(handle))
    assert len(saved) == 2
    assert {r["status"] for r in saved} == {"ok", "failed"}
    assert all(not r["PSNR"] for r in saved)
