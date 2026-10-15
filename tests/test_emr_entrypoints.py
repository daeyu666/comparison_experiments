"""Exercise real script imports without requiring a dataset, checkpoint or GPU."""
import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
EMR = ROOT / "comparison" / "EMR-Diff"


@pytest.mark.parametrize("script", ["test_hsi_deformed.py", "train_hsi_deformed.py"])
def test_emr_entrypoint_help_avoids_uafl_model_collision(script, tmp_path):
    # Running the actual entrypoint reproduces its own sys.path setup and
    # shared UAFL geometry imports. Importing from a clean unit-test process
    # alone used to miss the namespace-package collision.
    result = subprocess.run(
        [sys.executable, str(EMR / script), "--help"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "--dataset" in result.stdout
    assert "--checkpoint" in result.stdout or "--init_checkpoint" in result.stdout
