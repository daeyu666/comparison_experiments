"""Registered-only dispatch and non-RGB tensor compatibility."""
import importlib

import pytest
import torch

from experiment_protocol import build_test_command, build_train_command, resolve_model


@pytest.mark.parametrize("name", ["MoG-DCN", "ESFS"])
def test_registered_dispatch_and_mixed_rejection(name):
    adapter = resolve_model(name)
    train = build_train_command(adapter, "Chikusei", "registered", "cpu")
    test = build_test_command(adapter, "Chikusei", "registered", "cpu", check_files=False)
    assert train[1].endswith(f"/{name}/train.py")
    assert train[train.index("--scale_ratio") + 1] == "4"
    assert train[train.index("--validation_interval") + 1] == "5"
    assert test[1].endswith(f"/{name}/test.py")
    assert test[test.index("--test_mode") + 1] == "registered_only"
    with pytest.raises(ValueError, match="Registered-only"):
        build_train_command(adapter, "Chikusei", "mixed", "cpu")
    with pytest.raises(ValueError, match="Registered-only"):
        build_test_command(adapter, "Chikusei", "mixed", "cpu", check_files=False)


@pytest.mark.parametrize("name", ["MoG-DCN", "ESFS"])
def test_eight_band_msi_training_and_large_hsi_inference(name):
    module = importlib.import_module(f"comparison.{name}.model")
    saved_threads = torch.get_num_threads()
    torch.set_num_threads(min(3, saved_threads))
    try:
        torch.manual_seed(45)
        model = module.build_model({"n_bands": 128, "n_select_bands": 8, "scale_ratio": 4})
        lr, msi = torch.rand(1, 128, 16, 16), torch.rand(1, 8, 64, 64)
        prediction = model(lr, msi)
        assert prediction.shape == (1, 128, 64, 64)
        prediction.sub(torch.rand_like(prediction)).abs().mean().backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
        del model, prediction
        model = module.build_model({"n_bands": 242, "n_select_bands": 4, "scale_ratio": 4}).eval()
        with torch.inference_mode():
            prediction = model(torch.rand(1, 242, 32, 32), torch.rand(1, 4, 128, 128))
        assert prediction.shape == (1, 242, 128, 128)
        assert torch.isfinite(prediction).all()
    finally:
        torch.set_num_threads(saved_threads)
