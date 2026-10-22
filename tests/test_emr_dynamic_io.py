"""Observation-space EMR diffusion with a fixed-width feature backbone.

Run: python -m pytest -q tests/test_emr_dynamic_io.py
Synthetic random tensors only; no training/test dataset access is required.
"""
import sys
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
EMR = ROOT / "comparison" / "EMR-Diff"
sys.path.insert(0, str(EMR))

from arch.BAFUnet import BAFUNet
from EMRDiff import Edge
from model.ResShift_model import (
    EMR_ARCHITECTURE_ID,
    EMR_BACKBONE_WIDTH,
    ResShiftTrainer,
    save_checkpoint,
)


@pytest.fixture(autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(10)
    yield
    torch.set_num_threads(previous)


def trainer_for(hsi, msi, width=64):
    trainer = ResShiftTrainer.__new__(ResShiftTrainer)
    trainer.configs = OmegaConf.load(EMR / "config" / "5_step_EMRDiff.yaml")
    trainer.configs.train.hidden_channels = width
    trainer.hsi_channels = hsi
    trainer.msi_channels = msi
    trainer.state_channels = hsi + msi
    trainer.device = torch.device("cpu")
    trainer._apply_dynamic_channel_config()
    trainer.build_model()
    trainer.build_diffusion_model()
    return trainer


@pytest.mark.parametrize("hsi,msi", [(31, 3), (103, 4), (128, 8), (144, 8), (145, 8), (242, 4)])
@pytest.mark.parametrize("width", [64, 128])
def test_state_space_forward_multiscale_backward_and_reverse(hsi, msi, width):
    trainer = trainer_for(hsi, msi, width)
    net = trainer.Net
    assert isinstance(net, BAFUNet)
    assert net.in_channels == net.out_channels == hsi + msi
    assert net.model_channels == width
    assert trainer.backbone_width == width
    gt = torch.rand(2, hsi, 32, 32)
    lr = torch.rand(2, hsi, 8, 8)
    rgb = torch.rand(2, msi, 32, 32)
    condition = trainer._condition(lr, rgb, (32, 32))
    start = trainer._x_start(gt)
    t = torch.tensor([3, 3])
    noise = torch.randn_like(start)
    residual, edge = trainer.EMRDIFF.residual_calculator(start, condition, rgb)
    torch.testing.assert_close(residual, condition - start)
    noisy = trainer.EMRDIFF.forward_addnoise(start, condition, t, noise, rgb)
    assert noisy.shape == residual.shape == start.shape
    eta = float(trainer.EMRDIFF.etas[3])
    sigma = float(trainer.EMRDIFF.sqrt_etas[3] * trainer.EMRDIFF.kappa)
    torch.testing.assert_close(noisy, start + eta * (condition - start) + sigma * noise * edge)
    output, scales = net(noisy, rgb, condition[:, :hsi], t)
    assert output.shape == start.shape
    assert all(item.shape[1] == hsi + msi for item in scales)
    loss = trainer._multiscale_loss(output, scales, start, lr, rgb)
    assert torch.isfinite(loss)
    loss.backward()
    assert net.input_blocks_denoise[0][0].weight.grad is not None
    head = net.residual_heads[str(width)]
    assert head.weight.grad is not None
    assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
    # Exercise the original five-step reverse update in full spectral space.
    with torch.no_grad():
        edge_map = Edge()(rgb)
        current = trainer.EMRDIFF.prior_sample(condition, noise, edge_map)
        for step in reversed(range(5)):
            tt = torch.tensor([step, step])
            prediction, _ = net(current, rgb, condition[:, :hsi], tt)
            current = trainer.EMRDIFF.inverse_denoise(
                prediction + condition, current, tt,
                torch.randn_like(current), edge_map,
            )
            assert current.shape == start.shape
            assert torch.isfinite(current).all()


def test_internal_head_matches_previous_fixed64_external_head():
    trainer = trainer_for(103, 4)
    params = dict(trainer.configs.model.params)
    params["out_channels"] = 64
    core = BAFUNet(**params)
    core.load_state_dict({
        k: v for k, v in trainer.Net.state_dict().items()
        if not k.startswith("residual_heads.")
    }, strict=True)
    assert len(core.residual_heads) == 0
    x = torch.rand(1, 107, 32, 32)
    rgb = torch.rand(1, 4, 32, 32)
    lr_hr = torch.rand(1, 103, 32, 32)
    t = torch.tensor([2])
    with torch.no_grad():
        features, feature_scales = core(x, rgb, lr_hr, t)
        output, scales = trainer.Net(x, rgb, lr_hr, t)
        head = trainer.Net.residual_heads["64"]
        torch.testing.assert_close(output, head(features))
        for actual, feature in zip(scales, feature_scales):
            torch.testing.assert_close(actual, head(feature))


@pytest.mark.parametrize("width", [64, 128])
def test_checkpoint_roundtrip_and_old_architecture_rejection(tmp_path, width):
    trainer = trainer_for(103, 4, width)
    optimizer = torch.optim.AdamW(trainer.Net.parameters(), lr=1e-4)
    path = save_checkpoint(
        trainer.Net, optimizer, 1, "PaviaU", "physical", 107,
        checkpoint_dir=str(tmp_path),
    )
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert checkpoint["architecture_id"] == f"fixed{width}_dynamic_io_v2"
    assert checkpoint["backbone_width"] == width
    assert Path(path).name == ("last.pth.tar" if width == 64 else "last_base128.pth.tar")
    assert not list(tmp_path.glob("model_epoch_*.pth.tar"))
    trainer.verify_checkpoint_architecture(checkpoint)
    restored = trainer_for(103, 4, width)
    restored.Net.load_state_dict(checkpoint["model_state_dict"], strict=True)
    for architecture, width in [(None, 107), ("fixed34_state_projection_v1", 34), ("fixed64_state_projection_v1", 64)]:
        old = dict(checkpoint, architecture_id=architecture, backbone_width=width)
        with pytest.raises(ValueError, match="incompatible EMR-Diff architecture"):
            trainer.verify_checkpoint_architecture(old)
    with pytest.raises(ValueError, match="state_channels"):
        trainer.verify_checkpoint_architecture(dict(checkpoint, state_channels=136))
