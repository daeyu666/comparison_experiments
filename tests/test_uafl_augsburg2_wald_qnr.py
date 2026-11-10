"""Regression tests for the shared all-242-band HSI–MSI QNR."""
import numpy as np
import pytest

from comparison.UAFL.augsburg2_wald_qnr import hsi_msi_qnr, _windowed_uiqi_matrix


def test_uiqi_identity_is_one_for_constant_reflectance():
    x = np.full((48, 48, 1), 0.23, dtype=np.float32)
    result = _windowed_uiqi_matrix(x, x, np.ones(x.shape[:2], dtype=bool), 48)
    assert result[0, 0] == pytest.approx(1., abs=1e-9)


def test_qnr_identity_checks_all_hsi_pairs_and_only_msi_covered_pairs():
    rng = np.random.default_rng(19)
    lr = rng.uniform(0.12, 0.82, size=(16, 16, 242)).astype(np.float32)
    srfs = np.zeros((4, 242), dtype=np.float32)
    srfs[np.arange(4), [3, 40, 68, 110]] = 1
    hr = np.repeat(np.repeat(lr, 3, axis=0), 3, axis=1)
    msi = hr @ srfs.T
    mask = np.ones((48, 48), dtype=bool)
    result = hsi_msi_qnr(hr, lr, msi, mask, srfs)
    assert result["QNR"] == pytest.approx(1., abs=3e-5)
    assert result["spectral_pair_count"] == 242 * 241 // 2
    assert result["spatial_pair_count"] == 4
    assert not result["srf_projection_used"]
    # Alter an HSI band outside MSI coverage: spectral distortion responds
    # but the spatial term still uses only the four covered bands.
    distorted = hr.copy()
    distorted[..., -1] = 0
    result = hsi_msi_qnr(distorted, lr, msi, mask, srfs)
    assert result["Dlambda"] > 1e-5
    assert result["Ds"] == pytest.approx(0., abs=3e-5)


def test_qnr_rejects_incompatible_srf():
    h = np.zeros((48, 48, 242), dtype=np.float32)
    lr = np.zeros((16, 16, 242), dtype=np.float32)
    msi = np.zeros((48, 48, 4), dtype=np.float32)
    mask = np.ones((48, 48), dtype=bool)
    with pytest.raises(ValueError, match="Expected SRF"):
        hsi_msi_qnr(h, lr, msi, mask, np.ones((3, 242)))
