import math

import torch

from losses import SAMLoss


def test_sam_loss_identical_low_energy_spectra_are_zero():
    spectrum = torch.tensor([1e-4, 2e-4, 3e-4, 4e-4], dtype=torch.float32)
    target = spectrum.view(1, 4, 1, 1)
    pred = target.clone()
    assert SAMLoss()(pred, target).item() < 1e-6


def test_sam_loss_orthogonal_spectra_are_pi_over_two():
    pred = torch.tensor([[[[1.0]], [[0.0]]]], dtype=torch.float32)
    target = torch.tensor([[[[0.0]], [[1.0]]]], dtype=torch.float32)
    value = SAMLoss()(pred, target).item()
    assert math.isclose(value, math.pi / 2.0, abs_tol=1e-6)


def test_sam_loss_all_zero_pair_is_zero():
    pred = torch.zeros((1, 8, 2, 2), dtype=torch.float32)
    target = torch.zeros_like(pred)
    assert SAMLoss()(pred, target).item() == 0.0
