"""Exercise variant routing and the real stage-2 save loop without scene data."""
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiment_protocol as protocol
from emr_configuration import checkpoint_width, validate_hidden_channels

EMR = protocol.resolve_model("EMR-Diff")


def option(cmd, flag):
    return cmd[cmd.index(flag)+1]


def test_base128_train_resume_mixed_init_and_test_are_separate(tmp_path, monkeypatch):
    monkeypatch.setattr(protocol, "ROOT", tmp_path)
    stage1 = protocol.checkpoint_path(EMR, "PaviaU", "registered", 128)
    stage1.parent.mkdir(parents=True)
    stage1.touch()
    (stage1.parent / "last.pth.tar").touch()
    # Auto-resume must not select a co-located default-width checkpoint.
    with pytest.raises(FileNotFoundError, match="last_base128"):
        protocol.build_train_command(EMR, "PaviaU", "registered", "cuda:0", hidden_channels=128, resume="auto")
    (stage1.parent / "last_base128.pth.tar").touch()
    registered = protocol.build_train_command(EMR, "PaviaU", "registered", "cuda:0", hidden_channels=128, resume="auto")
    assert option(registered, "--hidden_channels") == "128"
    assert option(registered, "--resume").endswith("last_base128.pth.tar")
    assert Path(option(registered, "--checkpoint_dir")) == Path("comparison/EMR-Diff/checkpoints/physical/PaviaU")
    mixed = protocol.build_train_command(EMR, "PaviaU", "mixed", "cuda:0", hidden_channels=128)
    assert option(mixed, "--hidden_channels") == "128"
    assert option(mixed, "--init_checkpoint").endswith("best_base128.pth.tar")
    for mode in ("registered", "mixed"):
        cmd = protocol.build_test_command(EMR, "PaviaU", mode, "cuda:0", hidden_channels=128, check_files=False)
        assert option(cmd, "--checkpoint").endswith("best_base128.pth.tar")
        assert option(cmd, "--hidden_channels") == "128"
    assert "hidden_channels=128" in protocol.protocol_summary(EMR, hidden_channels=128)


def test_invalid_or_mismatched_widths_are_explicit():
    for value in (0, 31, 65, -128):
        with pytest.raises(ValueError):
            validate_hidden_channels(value)
    assert checkpoint_width({"backbone_width": 128}) == 128
    with pytest.raises(ValueError, match="differs"):
        checkpoint_width({"backbone_width": 64}, 128)
    with pytest.raises(ValueError, match="only to EMR-Diff"):
        protocol.build_train_command(protocol.resolve_model("CLSNet"), "PaviaU", "registered", "cpu", hidden_channels=128)


def test_stage2_actual_loop_saves_only_best_and_last_base128(tmp_path, monkeypatch):
    path = ROOT / "comparison" / "EMR-Diff" / "train_hsi_deformed.py"
    spec = importlib.util.spec_from_file_location("emr_width_stage2", path)
    stage2 = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage2)
    net = torch.nn.Linear(1, 1)
    p0 = SimpleNamespace(to=lambda device: p0, degrade=lambda gt: gt)

    class Loader:
        dataset = SimpleNamespace(degradation_operator=p0)

        def __iter__(self):
            yield {"gt": torch.ones(1, 1, 8, 8), "hr_msi": torch.ones(1, 1, 8, 8)}

    trainer = SimpleNamespace(Net=net, device=torch.device("cpu"), degradation_mode="physical",
                              dataset="PaviaU", state_channels=1, backbone_width=128,
                              architecture_id="fixed128_dynamic_io_v2", train_dataloader=Loader(),
                              validation_dataloader=[])
    monkeypatch.setattr(stage2, "build_trainer", lambda args: (trainer, tmp_path / "checkpoints", tmp_path / "logs"))
    monkeypatch.setattr(stage2, "validate_stage1_checkpoint", lambda *args: None)
    monkeypatch.setattr(stage2, "emr_training_loss", lambda *args: net.weight.square().mean())
    monkeypatch.setattr(stage2, "sample_training_geometry_batch", lambda *args, **kwargs: None)
    monkeypatch.setattr(stage2, "make_deformed_lr_hsi", lambda gt, geometry, degradation: gt)
    metrics = {"PSNR": 40.0, "SAM": 1.0, "SSIM": 0.99}
    monkeypatch.setattr(stage2, "evaluate_registered_and_warped", lambda *args: (metrics, metrics))
    monkeypatch.setattr(sys, "argv", [str(path), "--dataset", "PaviaU", "--device", "cpu",
                        "--hidden_channels", "128", "--epochs", "3", "--validation_interval", "2"])
    stage2.main()
    directory = tmp_path / "checkpoints"
    assert {p.name for p in directory.glob("*.pth.tar")} == {"best_base128.pth.tar", "last_base128.pth.tar"}
    saved = torch.load(directory / "last_base128.pth.tar", weights_only=True)
    assert saved["epoch"] == 2  # Stage-2's existing zero-based epoch convention.
    assert saved["backbone_width"] == 128
    assert saved["architecture_id"] == "fixed128_dynamic_io_v2"
