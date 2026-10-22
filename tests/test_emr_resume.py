"""Check stateful resume, old checkpoints and uninterrupted-training parity."""
import random
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "comparison" / "EMR-Diff"))
from model.ResShift_model import EMR_ARCHITECTURE_ID, ResShiftTrainer
from emr_configuration import architecture_id


class TinyNetwork(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, x, msi, lr, tt):
        return self.weight*x, []


class TinyDiffusion:
    def forward_addnoise(self, x_start, **kwargs):
        return x_start


def trainer_at(path, width=64):
    path.mkdir(parents=True, exist_ok=True)
    trainer = ResShiftTrainer.__new__(ResShiftTrainer)
    trainer.Net = TinyNetwork()
    trainer.optimizer = torch.optim.AdamW(trainer.Net.parameters(), lr=0.01, weight_decay=0.02)
    trainer.optimizer_name = "adamw"
    trainer.weight_decay = 0.02
    trainer.device = torch.device("cpu")
    trainer.architecture_id = architecture_id(width)
    trainer.backbone_width = width
    trainer.dataset, trainer.degradation_mode, trainer.state_channels = "PaviaU", "physical", 1
    trainer.diffusion_sf, trainer.num_timesteps = 4, 5
    trainer.checkpoint_dir = trainer.log_dir = str(path)
    trainer.configs = OmegaConf.create({"data": {
        "patch_size": 64, "stride": 32, "validation_size": 128, "test_size": 128,
        "mtf_nyquist": 0.2, "psf_truncate": 3.0,
    }})
    trainer.early_stop_metric, trainer.early_stop_min_delta, trainer.early_stop_patience = "PSNR", 0.02, 999999
    trainer.train_dataloader = [None]
    trainer.validation_dataloader = []
    trainer.EMRDIFF = TinyDiffusion()

    def prepare(_):
        value = (torch.rand(1, 1, 2, 2) + random.random() + np.random.random())/3
        return value, value, value

    trainer._prepare_batch = prepare
    trainer._x_start = lambda gt: gt
    trainer._condition = lambda lr, msi, hw: lr
    trainer._multiscale_loss = lambda output, scales, target, lr, msi: (output-target).square().mean()
    trainer.evaluate = lambda **kwargs: {"PSNR": 40.0, "SAM": 1.0}
    return trainer


def seed(value):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def test_resumed_updates_match_uninterrupted_training(tmp_path):
    seed(123)
    full = trainer_at(tmp_path / "full")
    full.train(4, 1)
    seed(123)
    partial = trainer_at(tmp_path / "split")
    partial.train(2, 1)
    seed(999)
    restored = trainer_at(tmp_path / "split")
    assert restored.resume_training(tmp_path / "split" / "last.pth.tar") == 2
    assert restored.resume_best_score == 40
    restored.train(4, 1)
    torch.testing.assert_close(full.Net.weight, restored.Net.weight, rtol=0, atol=0)
    full_state = full.optimizer.state_dict()["state"][0]
    restored_state = restored.optimizer.state_dict()["state"][0]
    for key in full_state:
        torch.testing.assert_close(full_state[key], restored_state[key], rtol=0, atol=0)
    assert full.optimizer.param_groups[0]["lr"] == restored.optimizer.param_groups[0]["lr"] == 0.01
    saved = torch.load(tmp_path / "split" / "last.pth.tar", weights_only=True)
    assert saved["epoch"] == 4
    assert saved["bad_evals"] == 3
    assert saved["best_epoch"] == 1
    best = torch.load(tmp_path / "split" / "best.pth.tar", weights_only=True)
    assert best["epoch"] == 1  # A worse resumed result must not replace the old best.
    lines = (tmp_path / "split" / "train_loss.csv").read_text().splitlines()
    assert [line.split(",")[0] for line in lines[1:]] == ["1", "2", "3", "4"]


def test_old_epoch800_checkpoint_resumes_at801_and_saves_final_epoch(tmp_path):
    seed(123)
    source = trainer_at(tmp_path / "source")
    source.train(1, 1)
    payload = torch.load(tmp_path / "source" / "last.pth.tar", weights_only=False)
    for key in ("training_stage", "optimizer_name", "best_epoch", "bad_evals", "rng_state", "training_protocol"):
        payload.pop(key)
    payload["epoch"], payload["best_score"] = 800, 50.0
    old = tmp_path / "model_epoch_800.pth.tar"
    torch.save(payload, old)
    resumed = trainer_at(tmp_path / "resumed")
    resumed.resume_training(old)
    assert resumed.start_epoch == 800
    assert resumed.optimizer.state_dict()["state"][0]["step"] == 1
    with pytest.raises(ValueError, match="must exceed"):
        resumed.train(800, 20)
    resumed.train(801, 20)
    saved = torch.load(tmp_path / "resumed" / "last.pth.tar", weights_only=True)
    assert saved["epoch"] == 801
    assert saved["best_score"] == 50.0
    assert saved["optimizer_state_dict"]["state"][0]["step"] == 2
    lines = (tmp_path / "resumed" / "train_loss.csv").read_text().splitlines()
    assert lines[1].startswith("801,")


@pytest.mark.parametrize("width", [64, 128])
def test_training_keeps_best_and_last_and_last_tracks_nonvalidation_epoch(tmp_path, monkeypatch, width):
    from model import ResShift_model as module
    calls = []
    original = module.save_checkpoint

    def record(*args, **kwargs):
        path = original(*args, **kwargs)
        calls.append((args[2], Path(path).name))
        return path

    monkeypatch.setattr(module, "save_checkpoint", record)
    trainer = trainer_at(tmp_path, width)
    trainer.train(3, 2)
    tag = "" if width == 64 else "_base128"
    assert calls == [(1, f"last{tag}.pth.tar"), (2, f"best{tag}.pth.tar"),
                     (2, f"last{tag}.pth.tar"), (3, f"last{tag}.pth.tar")]
    assert {p.name for p in tmp_path.glob("*.pth.tar")} == {f"best{tag}.pth.tar", f"last{tag}.pth.tar"}
    saved = torch.load(tmp_path / f"last{tag}.pth.tar", weights_only=True)
    assert saved["epoch"] == 3
    assert saved["architecture_id"] == architecture_id(width)
    resumed = trainer_at(tmp_path, width)
    resumed.resume_training(tmp_path / f"last{tag}.pth.tar")
    assert resumed.start_epoch == 3
    if width == 128:
        with pytest.raises(ValueError, match="incompatible EMR-Diff architecture"):
            trainer_at(tmp_path / "base64").resume_training(tmp_path / "last_base128.pth.tar")


@pytest.mark.parametrize("change,expected", [
    ({"training_stage": "stage2_registered_deformed_mixed"}, "registered checkpoint"),
    ({"dataset": "CAVE"}, "dataset mismatch"),
    ({"optimizer_name": "adam"}, "optimizer mismatch"),
    ({"training_protocol": {"scale_ratio": 3}}, "protocol mismatch"),
])
def test_resume_rejects_incompatible_training_state(tmp_path, change, expected):
    source = trainer_at(tmp_path / "source")
    source.train(1, 1)
    state = torch.load(tmp_path / "source" / "last.pth.tar", weights_only=False)
    state.update(change)
    path = tmp_path / "bad.pth.tar"
    torch.save(state, path)
    target = trainer_at(tmp_path / "target")
    with pytest.raises(ValueError, match=expected):
        target.resume_training(path)
