"""Protocol, checkpoint and observation helpers for CLSNet."""
from __future__ import annotations

import hashlib
import random
from pathlib import Path

import numpy as np
import torch

from comparison.UAFL.common import build_shared_cfg, require_srf_weights, set_seed
from comparison.UAFL.hsi_deformation import (
    make_deformed_lr_hsi, sample_synthetic_geometry, sample_training_geometry_batch,
)
from comparison.CLSNet.model import ARCHITECTURE_ID, UPSTREAM_COMMIT, build_clsnet
from experiment_protocol import DATASETS, DATASET_VALIDATION_INTERVAL

HERE = Path(__file__).resolve().parent
PROTOCOL_KEYS = (
    'dataset', 'scale_ratio', 'degradation_mode', 'mtf_nyquist', 'psf_truncate',
    'degradation_sigma', 'degradation_kernel_size', 'msi_mode', 'srf_path',
    'wavelength_path', 'wavelength_root', 'srf_interp', 'srf_band_set',
    'image_size', 'patch_size', 'stride',
)
GEOMETRY_KEYS = ('max_translation', 'max_rotation_deg', 'max_local_px',
                 'control_grid', 'min_jacobian', 'registered_probability')


def add_data_args(p):
    p.add_argument('--dataset', choices=DATASETS, default='PaviaU')
    p.add_argument('--data_root', default='./data/raw')
    for name, default in [('image_size',128), ('patch_size',64), ('stride',32), ('scale_ratio',4)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--degradation_mode', choices=['physical','gaussian_bicubic'], default='physical')
    p.add_argument('--degradation_sigma', type=float, default=2.)
    p.add_argument('--degradation_kernel_size', type=int, default=5)
    p.add_argument('--mtf_nyquist', type=float, default=.2)
    p.add_argument('--psf_truncate', type=float, default=3.)
    p.add_argument('--msi_mode', choices=['srf'], default='srf')
    p.add_argument('--srf_path', default='')
    p.add_argument('--wavelength_root', default='./data/wavelengths')
    p.add_argument('--wavelength_path', default='')
    p.add_argument('--srf_interp', choices=['pchip','linear'], default='pchip')
    p.add_argument('--srf_band_set', default='auto')
    p.add_argument('--device', default='cuda:0')


def add_geometry_args(p):
    p.add_argument('--max_translation', type=float, default=4.)
    p.add_argument('--max_rotation_deg', type=float, default=2.)
    p.add_argument('--max_local_px', type=float, default=4.)
    p.add_argument('--control_grid', type=int, default=5)
    p.add_argument('--min_jacobian', type=float, default=.5)
    p.add_argument('--registered_probability', type=float, default=.10)


def resolve_device(name):
    if str(name).startswith('cuda') and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; use --device cpu for a CPU check')
    return torch.device(name)


def make_generator(device, seed):
    return torch.Generator(device=device).manual_seed(int(seed))


def model_for_info(info, args, device):
    require_srf_weights(info)
    model = build_clsnet(info['n_bands'], info['n_select_bands'], args.scale_ratio).to(device)
    model.sensor_fingerprint = hashlib.sha256(
        require_srf_weights(info).cpu().numpy().tobytes()).hexdigest()
    return model


def snapshot_rng(geometry_gen=None):
    return dict(python=random.getstate(), numpy=np.random.get_state(),
                torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                geometry=geometry_gen.get_state() if geometry_gen is not None else None)


def restore_rng(state, geometry_gen=None):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state.get('cuda') is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([x.cpu() for x in state['cuda']])
    if geometry_gen is not None and state.get('geometry') is not None:
        geometry_gen.set_state(state['geometry'].cpu())


def checkpoint_payload(model, optimizer, epoch, best, bad, args, generator):
    return dict(method='CLSNet', architecture_id=ARCHITECTURE_ID,
                upstream_commit=UPSTREAM_COMMIT, sensor_fingerprint=getattr(model,'sensor_fingerprint',None),
                hsi_channels=model.hsi_channels,
                msi_channels=model.msi_channels, backbone_width=62,
                training_stage=args.stage,
                registered_probability=(1.0 if args.stage=='registered' else args.registered_probability),
                args=vars(args).copy(),
                model=model.state_dict(), optimizer=optimizer.state_dict(),
                epoch=epoch, best_psnr=best, bad_validations=bad,
                rng=snapshot_rng(generator))


def verify_checkpoint(checkpoint, model, args, *, expected_stage):
    if checkpoint.get('method') != 'CLSNet' or checkpoint.get('architecture_id') != ARCHITECTURE_ID:
        raise ValueError('CLSNet checkpoint architecture/method mismatch')
    if checkpoint.get('hsi_channels') != model.hsi_channels or checkpoint.get('msi_channels') != model.msi_channels:
        raise ValueError('CLSNet checkpoint observation channels mismatch')
    if checkpoint.get('training_stage') != expected_stage:
        raise ValueError(f'CLSNet checkpoint stage mismatch: expected {expected_stage}')
    if checkpoint.get('backbone_width') != 62:
        raise ValueError('CLSNet checkpoint backbone width mismatch')
    fingerprint = getattr(model,'sensor_fingerprint',None)
    if fingerprint is not None and checkpoint.get('sensor_fingerprint') != fingerprint:
        raise ValueError('CLSNet checkpoint sensor SRF mismatch')
    recorded = checkpoint.get('args', {})
    for key in PROTOCOL_KEYS + (GEOMETRY_KEYS if expected_stage == 'mixed' else ()):
        if hasattr(args, key) and recorded.get(key) != getattr(args, key):
            raise ValueError(f'CLSNet checkpoint protocol mismatch: {key}: {recorded.get(key)!r} != {getattr(args,key)!r}')


def load_checkpoint(path, model, args, *, expected_stage, optimizer=None):
    checkpoint = torch.load(path, map_location=next(model.parameters()).device, weights_only=False)
    verify_checkpoint(checkpoint, model, args, expected_stage=expected_stage)
    model.load_state_dict(checkpoint['model'], strict=True)
    if optimizer is not None:
        if checkpoint.get('optimizer') is None or 'rng' not in checkpoint:
            raise ValueError('Resume requires optimizer and RNG states')
        # LR is deliberately kept at the recorded value for exact continuation.
        for key in ('lr','weight_decay','batch_size','seed','validation_cases','validation_interval'):
            if checkpoint['args'].get(key) != getattr(args,key):
                raise ValueError(f'Resume schedule mismatch: {key}')
        optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint


def geometry_kwargs(args):
    return {name:getattr(args,name) for name in GEOMETRY_KEYS if name != 'registered_probability'}


def deformed_observation(gt, p0, args, generator, *, mixed):
    """Warp HR-HSI before P0; exact identity samples are never resampled/warped."""
    if not mixed:
        return p0.degrade(gt), gt.shape[0]
    identity = torch.rand(gt.shape[0], device=gt.device, generator=generator) < args.registered_probability
    if bool(identity.all()):
        return p0.degrade(gt), int(identity.sum())
    geometry = sample_training_geometry_batch(
        gt.shape[0], *gt.shape[-2:], device=gt.device, dtype=gt.dtype,
        generator=generator, **geometry_kwargs(args))
    warped = make_deformed_lr_hsi(gt, geometry, p0)
    if bool(identity.any()):
        warped = torch.where(identity[:,None,None,None], p0.degrade(gt), warped)
    return warped, int(identity.sum())
