"""Exercise final-test geometry and provenance with a small CPU stand-in model."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "comparison" / "EMR-Diff" / "test_hsi_deformed.py"


def test_final_evaluation_records_reused_geometry_and_checkpoint_stage(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("emr_final_evaluation", SCRIPT)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    checkpoint_path = tmp_path / "best.pth.tar"
    checkpoint_path.touch()
    output = tmp_path / "result.json"
    monkeypatch.setattr(sys, "argv", [
        str(SCRIPT), "--dataset", "CAVE", "--device", "cpu",
        "--cases", "2", "--checkpoint", str(checkpoint_path),
        "--output_json", str(output),
    ])

    class Degradation:
        def to(self, device):
            return self

        def degrade(self, image):
            return F.avg_pool2d(image, 4)

    class TestSet:
        degradation_operator = Degradation()

        def __len__(self):
            return 2

        def __getitem__(self, index):
            ramp = torch.linspace(0.1, 0.8, 128).repeat(128, 1)
            gt = torch.stack([ramp, ramp.T, (ramp + ramp.T) / 2])
            return {"gt": gt, "hr_msi": gt[:1] * (index + 1) / 2}

    test_set = TestSet()
    msi_inputs = []

    def reconstruct(lr_hsi, hr_msi):
        msi_inputs.append(hr_msi.clone())
        return F.interpolate(lr_hsi, scale_factor=4, mode="bilinear", align_corners=False)

    trainer = SimpleNamespace(
        device=torch.device("cpu"), test_dataloader=SimpleNamespace(dataset=test_set),
        Net=torch.nn.Module(), state_channels=4, _reconstruct=reconstruct,
        verify_checkpoint_architecture=lambda *args, **kwargs: None,
    )
    checkpoint = {
        "dataset": "CAVE", "state_channels": 4, "degradation_mode": "physical",
        "training_stage": "stage2_registered_deformed_mixed",
        "registered_probability": 0.1, "architecture_id": "stand_in",
        "epoch": 17, "model_state_dict": {},
    }
    monkeypatch.setattr(evaluator, "build_trainer", lambda args: trainer)
    monkeypatch.setattr(evaluator.torch, "load", lambda *args, **kwargs: checkpoint)
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        evaluator.main()
    finally:
        torch.set_num_threads(previous_threads)

    result = json.loads(output.read_text())
    conditions = result["test_conditions"]
    assert conditions["checkpoint_training_stage"] == checkpoint["training_stage"]
    assert conditions["checkpoint_epoch"] == 17
    assert conditions["test_patch_count"] == conditions["registered_observations"] == 2
    assert conditions["warp_observations"] == 4
    assert conditions["geometry_generator_seed"] == 70010
    assert conditions["local_strength_min_fraction"] == 0.0
    rows = result["per_sample_case"]
    assert len(rows) == 4
    summary = result["geometry_summary"]
    assert summary["distinct_geometry_cases"] == 2
    for key in ("dx_hr_px", "dy_hr_px", "rotation_deg", "local_max_hr_px", "min_jacobian"):
        assert [r[key] for r in rows[:2]] == [r[key] for r in rows[2:]]
        values = [r[key] for r in rows[:2]]
        assert summary[key] == {"min": min(values), "mean": sum(values) / 2, "max": max(values)}
    assert all(0 <= r["local_max_hr_px"] <= 4.0001 for r in rows)
    assert all(r["min_jacobian"] >= 0.5 for r in rows)
    for index, msi in enumerate(msi_inputs):
        torch.testing.assert_close(msi.squeeze(0), test_set[index // 3]["hr_msi"])
    assert result["average_metrics"]["Registered"]["PSNR"] > result["average_metrics"]["Warp"]["PSNR"]
    assert evaluator.summarize_geometry_cases([]) == {}
