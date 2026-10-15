"""Exercise the root launcher and checkpoint adapter guards without real data."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
spec = importlib.util.spec_from_file_location("comparison_fixed_entry", ROOT / "test_fixed_deformation.py")
entry = importlib.util.module_from_spec(spec)
spec.loader.exec_module(entry)


def test_root_dispatch_runs_selected_level_on_all_patches(tmp_path, monkeypatch):
    checkpoint = tmp_path / "best.pth.tar"
    checkpoint.touch()
    out = tmp_path / "results"
    test_set = [{"gt": torch.rand(2, 32, 32), "hr_msi": torch.rand(1, 32, 32)} for _ in range(2)]

    def builder(args):
        assert args.mode == "registered"
        assert args.checkpoint == str(checkpoint)
        assert args.image_size == 128
        return (test_set, lambda lr, msi: F.interpolate(lr, scale_factor=4),
                lambda x: F.avg_pool2d(x, 4), torch.device("cpu"), {"training_stage": "registered"})

    monkeypatch.setitem(entry.MODEL_BUILDERS, "EMR-Diff", builder)
    monkeypatch.setattr(sys, "argv", ["test_fixed_deformation.py", "--model", "EMR-Diff",
                        "--dataset", "CAVE", "--mode", "registered", "--device", "cpu",
                        "--checkpoint", str(checkpoint), "--output_dir", str(out), "--cases", "1", "--levels", "6"])
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        entry.main()
    finally:
        torch.set_num_threads(previous)
    result = json.loads((out / "summary.json").read_text())
    assert result["test_conditions"]["model"]["training_stage"] == "registered"
    assert len(result["results"]) == 1
    assert result["results"][0]["level"] == 6
    assert result["results"][0]["test_patch_count"] == 2


@pytest.mark.parametrize("mode,stage,allowed", [
    ("registered", None, True), ("registered", "stage2_registered_deformed_mixed", False),
    ("mixed", "stage2_registered_deformed_mixed", True), ("mixed", None, False),
])
def test_emr_adapter_checks_weight_stage(mode, stage, allowed, tmp_path, monkeypatch):
    test_set = SimpleNamespace(degradation_operator=SimpleNamespace(to=lambda device: SimpleNamespace(degrade=lambda x: x)))
    trainer = SimpleNamespace(
        device=torch.device("cpu"), state_channels=34, Net=torch.nn.Module(),
        test_dataloader=SimpleNamespace(dataset=test_set), _reconstruct=lambda x, y: x,
        verify_checkpoint_architecture=lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(entry, "_load_evaluator", lambda folder: SimpleNamespace(build_trainer=lambda args: trainer))
    path = tmp_path / "checkpoint.pth"
    torch.save({"model_state_dict": {}, "dataset": "CAVE", "state_channels": 34,
                "degradation_mode": "physical", "training_stage": stage}, path)
    args = SimpleNamespace(checkpoint=str(path), dataset="CAVE", mode=mode)
    if allowed:
        assert entry._emr(args)[0] is test_set
    else:
        with pytest.raises(ValueError, match="requires"):
            entry._emr(args)


@pytest.mark.parametrize("model", ["EMR-Diff", "UAFL"])
def test_dry_run_entrypoint_needs_no_dataset_or_checkpoint(model, tmp_path):
    result = subprocess.run(
        [sys.executable, str(ROOT / "test_fixed_deformation.py"), "--model", model,
         "--dataset", "CAVE", "--dry_run"],
        cwd=tmp_path, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert '"level": 6' in result.stdout
    assert '"translation_hr_px": 6.0' in result.stdout
