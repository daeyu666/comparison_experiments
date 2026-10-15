"""Use the same observed Wald cache, masks and heldout rules as EMR-Diff/UAFL."""
import torch
from comparison.UAFL.augsburg2_wald_common import (
    WaldDataset, PROVENANCE, correct_msi, make_loaders, masked_l1,
    metrics_from_sums, metrics_sums, read_json, read_radiometry, require_wald,
)
from comparison.CLSNet.model import verify_architecture


def predict_clsnet(model, lr_hsi, hr_msi):
    # CLSNet owns nearest upsampling and paired padding; never use UAFL bicubic.
    return model(lr_hsi, hr_msi)


def load_state(path, model, optimizer=None, device='cpu'):
    state = torch.load(path,map_location=device,weights_only=False)
    if (state.get('protocol') != 'Augsburg-2-Wald-CLSNet'
        or state.get('msi_source') != PROVENANCE
        or state.get('scale_ratio') != 3
        or state.get('split_protocol_id') != 'Augsburg2-Wald-center-holdout-v1'
        or state.get('hsi_channels') != 242 or state.get('msi_channels') != 4):
        raise ValueError('Require center-heldout Wald x3 CLSNet checkpoint')
    verify_architecture(state, model)
    model.load_state_dict(state['model'],strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state['optimizer'])
    return state
