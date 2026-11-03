"""Shared benchmark configuration helpers for IR&ArF."""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from comparison.CLSNet.common import (
    add_data_args, add_geometry_args, build_shared_cfg,
    deformed_observation, make_generator, resolve_device, set_seed,
    snapshot_rng, restore_rng, geometry_kwargs,
    sample_synthetic_geometry, make_deformed_lr_hsi, require_srf_weights,
)
from experiment_protocol import DATASET_VALIDATION_INTERVAL

__all__ = [
    "HERE", "ROOT", "add_data_args", "add_geometry_args", "build_shared_cfg",
    "deformed_observation", "make_generator", "resolve_device", "set_seed",
    "snapshot_rng", "restore_rng", "geometry_kwargs", "sample_synthetic_geometry",
    "make_deformed_lr_hsi", "require_srf_weights", "DATASET_VALIDATION_INTERVAL",
]
