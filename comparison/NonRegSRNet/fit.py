"""Deterministic pairwise optimization, optional resumable observation cache."""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import torch

from .model import NonRegSRNet, FitConfig, ARCHITECTURE_ID, model_metadata


def observation_key(lr_hsi, hr_msi, srf, scale, config):
    digest = hashlib.sha256()
    meta = dict(architecture_id=ARCHITECTURE_ID, scale=int(scale), config=asdict(config))
    digest.update(json.dumps(meta, sort_keys=True).encode())
    for tensor in (lr_hsi, hr_msi, srf):
        value = tensor.detach().cpu().float().contiguous()
        digest.update(str(tuple(value.shape)).encode())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def _atomic_save(state, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def fit_pair(lr_hsi, hr_msi, srf, scale=4, config=None, *, cache_dir=None,
             save_interval=500, progress=False):
    """Fit one independent observation pair. Neither GT nor deformation truth is accepted."""
    config = (config or FitConfig()).validate()
    if lr_hsi.shape[0] != 1 or hr_msi.shape[0] != 1:
        raise ValueError("Official observation-wise protocol uses one pair at a time")
    if lr_hsi.device != hr_msi.device:
        raise ValueError("HSI and MSI must use the same device")
    if not torch.isfinite(lr_hsi).all() or not torch.isfinite(hr_msi).all():
        raise ValueError("Observations must be finite")
    device = lr_hsi.device
    devices = []
    if device.type == "cuda":
        devices = [device.index if device.index is not None else torch.cuda.current_device()]
    key = observation_key(lr_hsi, hr_msi, srf, scale, config)
    cache_file = Path(cache_dir)/(key+".pth.tar") if cache_dir else None
    # Shared benchmark scorers use no_grad. Pair adaptation must still backprop;
    # restore caller RNG so optimization never changes the geometry-case schedule.
    with torch.inference_mode(False), torch.enable_grad(), torch.random.fork_rng(devices=devices):
        torch.random.default_generator.manual_seed(config.seed)
        if devices:
            torch.cuda.default_generators[devices[0]].manual_seed(config.seed)
        lr = lr_hsi.detach().clone().float()
        msi = hr_msi.detach().clone().float()
        weights = torch.as_tensor(srf).detach().clone().float()
        model = NonRegSRNet(lr.shape[1], msi.shape[1], weights, scale, config.num_rho).to(device)
        optimizer = model.make_optimizer(config)

        def multiplier(epoch):
            # Same LambdaLR function and epoch_count=1 as released network.py.
            return max(0.0, 1.0-max(0, epoch+2-config.constant_steps)/float(config.decay_steps+1))

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=multiplier)
        start = 0
        last_loss = None
        if cache_file is not None and cache_file.is_file():
            state = torch.load(cache_file, map_location="cpu", weights_only=False)
            if (state.get("architecture_id") != ARCHITECTURE_ID
                or state.get("observation_sha256") != key
                or state.get("fit_config") != asdict(config)):
                raise ValueError("Pair cache observation/configuration mismatch")
            model.load_state_dict(state["model"], strict=True)
            optimizer.load_state_dict(state["optimizer"])
            scheduler.load_state_dict(state["scheduler"])
            start = int(state["completed_iterations"])
            last_loss = state.get("last_loss")
            if not 0 <= start <= config.iterations:
                raise ValueError("Invalid pair-cache optimization step")
        model.train()
        iterator = range(start, config.iterations)
        if progress:
            from tqdm import tqdm
            iterator = tqdm(iterator, total=config.iterations, initial=start,
                            desc="NonRegSRNet pair fit", dynamic_ncols=True,
                            mininterval=1.0, disable=None)

        def save(completed):
            if cache_file is not None:
                _atomic_save({
                    **model_metadata(model,config), "observation_sha256":key,
                    "completed_iterations":completed, "last_loss":last_loss,
                    "model":model.state_dict(), "optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),
                    "gt_used_for_optimization":False,
                }, cache_file)

        for step in iterator:
            optimizer.zero_grad(set_to_none=True)
            loss, terms = model.observation_loss(lr, msi, config)
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"NonRegSRNet nonfinite observation loss at iteration {step+1}")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float("inf"), error_if_nonfinite=True)
            optimizer.step()
            model.project_parameters()
            scheduler.step()
            last_loss = float(loss.detach())
            if progress:
                iterator.set_postfix(loss=f"{last_loss:.5g}", refresh=False)
            if save_interval > 0 and (step+1)%save_interval == 0:
                save(step+1)
        if start < config.iterations:
            save(config.iterations)
        with torch.no_grad():
            prediction = model.reconstruct(msi).detach()
        report = {
            **model_metadata(model,config), "observation_sha256":key,
            "completed_iterations":config.iterations, "resumed_from_iteration":start,
            "last_loss":last_loss, "gt_used_for_optimization":False,
        }
        return prediction, report


class PairwisePredictor:
    """Shared suite callback: every changed LR-HSI observation gets its own fit."""
    def __init__(self, srf, scale=4, config=None, cache_dir=None):
        self.srf = torch.as_tensor(srf).detach().clone().cpu()
        self.scale, self.config = int(scale), (config or FitConfig()).validate()
        self.cache_dir = cache_dir
        self.last_report = None

    def __call__(self, lr_hsi, hr_msi):
        prediction, self.last_report = fit_pair(
            lr_hsi,hr_msi,self.srf,self.scale,self.config,cache_dir=self.cache_dir)
        return prediction
