"""CLI tests for the shared multiseed main-table entrypoint."""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiment_protocol import resolve_model
from test_fixed_deformation import MODEL_BUILDERS
from test_multiseed_deformation import parse_args


@pytest.mark.parametrize("model", ["UAFL", "EMR-Diff", "CLSNet", "DRTNet"])
@pytest.mark.parametrize("dataset", ["PaviaU", "CAVE", "Botswana", "Augsburg"])
def test_model_and_dataset_are_selectable(model, dataset):
    args = parse_args(["--model", model, "--dataset", dataset, "--dry_run"])
    adapter = resolve_model(args.model)
    assert adapter.canonical == model
    assert adapter.canonical in MODEL_BUILDERS
    assert args.dataset == dataset


def test_default_main_table_uses_ten_geometry_seeds_and_ten_cases():
    args = parse_args(["--model", "UAFL", "--dataset", "PaviaU", "--dry_run"])
    assert args.geometry_seeds == list(range(10001, 10011))
    assert args.cases == 10
    assert args.eval_seed == 1234
    assert args.max_samples == 0
    assert args.scale_ratio == 4
    assert args.mtf_nyquist == pytest.approx(0.2)
    assert args.psf_truncate == pytest.approx(3.0)
