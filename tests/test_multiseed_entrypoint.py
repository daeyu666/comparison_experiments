"""Check root model/dataset switching and all three shared test adapters."""
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_protocol as protocol
import test_multiseed_deformation as entry


@pytest.mark.parametrize("name", ["EMR-Diff", "UAFL", "CLSNet"])
def test_mixed_root_command_preserves_model_dataset_switching(name, tmp_path):
    command = protocol.build_test_command(protocol.resolve_model(name), "CAVE", "mixed", "cpu", check_files=False)
    assert Path(command[1]).name == "test_multiseed_deformation.py"
    args = entry.parse_args(command[2:])
    assert args.model == name and args.dataset == "CAVE" and args.mode == "mixed"
    assert args.geometry_seeds == list(range(10001, 10011)) and args.cases == 10
    old = protocol.build_test_command(protocol.resolve_model(name), "CAVE", "mixed", "cpu", check_files=False, legacy_single_seed=True)
    assert Path(old[1]).name == "test_hsi_deformed.py"
    registered = protocol.build_test_command(protocol.resolve_model(name), "CAVE", "registered", "cpu", check_files=False)
    assert Path(registered[1]).name == "test_hsi_deformed.py"
    assert registered[registered.index("--test_mode") + 1] == "registered_only"


@pytest.mark.parametrize("name", ["EMR-Diff", "UAFL", "CLSNet"])
def test_selected_adapter_receives_frozen_protocol_and_visits_all_cases(name, tmp_path, monkeypatch):
    checkpoint = tmp_path / "mixed.pth"
    checkpoint.write_bytes(b"test fixture")
    output = tmp_path / "out"
    data = [{"gt": torch.rand(2, 32, 32), "hr_msi": torch.rand(1, 32, 32)} for _ in range(2)]
    calls = []
    def build(args):
        assert args.mode == "mixed" and args.image_size == 128
        assert args.registered_probability == 0.1 and args.max_local_px == 4
        assert args.checkpoint == str(checkpoint)
        def predict(lr, msi):
            calls.append(1)
            return F.interpolate(lr, scale_factor=4)
        return data, predict, lambda x: F.avg_pool2d(x, 4), "cpu", {"training_stage": "mixed"}
    monkeypatch.setitem(entry.MODEL_BUILDERS, name, build)
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        result = entry.main(["--model", name, "--dataset", "CAVE", "--device", "cpu",
            "--checkpoint", str(checkpoint), "--output_dir", str(output),
            "--geometry_seeds", "101", "202", "--cases", "2"])
    finally:
        torch.set_num_threads(previous)
    assert len(calls) == 2 + 2 * 2 * 2
    assert result["complete"] and result["test_conditions"]["model"]["method"] == name


def test_root_dry_run_needs_no_checkpoint_and_registered_stays_single():
    for mode in ("registered", "mixed"):
        result = subprocess.run([sys.executable, str(ROOT / "test.py"), "--model", "CLSNet",
                                 "--dataset", "CAVE", "--mode", mode, "--dry_run"],
                                capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stdout + result.stderr
        if mode == "mixed":
            assert "test_multiseed_deformation.py" in result.stdout
            assert "--geometry_seeds 10001 10002" in result.stdout
        else:
            assert "--test_mode registered_only" in result.stdout


def test_real_clsnet_adapter_loads_mixed_weights_and_uses_new_protocol(tmp_path, monkeypatch):
    from comparison.CLSNet.common import checkpoint_payload, make_generator, model_for_info
    import data_loader
    info = {"n_bands": 31, "n_select_bands": 3, "srf_weights": torch.full((3, 31), 1/31)}
    class P0:
        def to(self, device):
            return self
        def degrade(self, x):
            return F.avg_pool2d(x, 4)
    class Dataset:
        degradation_operator = P0()
        def __len__(self):
            return 1
        def __getitem__(self, index):
            gt = torch.rand(31, 24, 24, generator=torch.Generator().manual_seed(101))
            return {"gt": gt, "hr_msi": gt[:3]}
    path = tmp_path / "clsnet_mixed.pth"
    argv = ["--model", "CLSNet", "--dataset", "CAVE", "--device", "cpu",
            "--checkpoint", str(path), "--output_dir", str(tmp_path / "out"),
            "--geometry_seeds", "101", "202", "--cases", "1"]
    args = entry.parse_args(argv)
    args.stage = "mixed"
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        model = model_for_info(info, args, "cpu")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
        torch.save(checkpoint_payload(model, optimizer, 1, 1.0, 0, args, make_generator(torch.device("cpu"), 10)), path)
        monkeypatch.setattr(data_loader, "build_datasets", lambda *a, **kw: (None, None, Dataset(), info))
        result = entry.main(argv)
    finally:
        torch.set_num_threads(previous)
    assert result["complete"] and result["Warp"]["n_geometry_seeds"] == 2
    assert result["test_conditions"]["model"]["backbone_width"] == 62
