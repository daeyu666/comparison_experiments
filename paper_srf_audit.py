"""Verify that an archived checkpoint used the shared qualitative sensor."""
from __future__ import annotations

import hashlib
from itertools import product

import numpy as np

from srf_utils import build_srf_weights


SRF_ARGS = ("msi_mode", "srf_band_set", "srf_interp", "srf_path", "wavelength_path", "wavelength_root")


def fingerprint(weights):
    return hashlib.sha256(np.asarray(weights, dtype=np.float32).tobytes()).hexdigest()


def differences(a, b):
    delta = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {"max_abs": float(delta.max()), "max_band_l1": float(delta.sum(axis=1).max())}


def rounding_variants(raw):
    """Reconstruct possible PaviaU matrices with a slightly different float32 reduction."""
    choices = []
    for row in raw:
        total = row.sum()
        low = high = total
        denominators = [total, np.float32(row.sum(dtype=np.float64))]
        for _ in range(2):
            low = np.nextafter(low, np.float32(-np.inf))
            high = np.nextafter(high, np.float32(np.inf))
            denominators.extend((low, high))
        rows = [(row / value).tobytes() for value in denominators]
        rows.append((row.astype(np.float64) / row.sum(dtype=np.float64)).astype(np.float32).tobytes())
        choices.append(tuple(dict.fromkeys(rows)))
    for rows in product(*choices):
        yield np.frombuffer(b"".join(rows), dtype=np.float32).reshape(raw.shape)


def audit(checkpoint, shared, cfg, source=None):
    """Recover the exact archived hash before accepting a numerically equivalent SRF."""
    from data_loader import _resolve_srf_spec

    expected = checkpoint.get("sensor_fingerprint")
    current = fingerprint(shared)
    recorded = checkpoint.get("args") or {}
    report = {"checkpoint_sha256": expected, "shared_sha256": current, "shape": list(shared.shape),
              "training_srf_args": {k: recorded[k] for k in SRF_ARGS if k in recorded}}

    def match(candidate, label):
        if candidate.shape != shared.shape or not np.isfinite(candidate).all() or np.any(candidate < 0):
            return None
        if fingerprint(candidate) != expected:
            return None
        diff = differences(candidate, shared)
        # For X in [0,1], the row L1 difference also bounds the absolute MSI change.
        status = "roundoff" if diff["max_band_l1"] <= 1e-6 else "different_sensor"
        return dict(report, source=label, status=status, **diff)

    if expected == current:
        return dict(report, status="exact", source="shared", **differences(shared, shared))
    if source:
        weights = np.asarray(np.load(source, allow_pickle=False), dtype=np.float32)
        result = match(weights, "training_matrix")
        if result:
            return result
        return dict(report, status="unverified", source="training_matrix",
                    provided_sha256=fingerprint(weights), provided_shape=list(weights.shape))

    path, bands, wavelengths, _, _ = _resolve_srf_spec(cfg, shared.shape[1])
    other = "linear" if cfg.srf_interp == "pchip" else "pchip"
    alternative, _ = build_srf_weights(path, wavelengths, bands, interp_kind=other)
    result = match(alternative, other)
    if result:
        return result
    if cfg.dataset == "PaviaU" and shared.shape == (4, 103):
        raw, _ = build_srf_weights(path, wavelengths, bands, interp_kind=cfg.srf_interp, normalize=False)
        for candidate in rounding_variants(raw):
            result = match(candidate, "float32_normalization")
            if result:
                return result
    return dict(report, status="unverified", alternative_interp=other,
                alternative_sha256=fingerprint(alternative))
