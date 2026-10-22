import glob
import os
import random

import numpy as np
import scipy.io as sio
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from arch.BAFUnet import BAFUNet
from dataset_loader.ufg_adapter import build_ufg_loaders
from EMRDiff import EMRDIFF, Edge
from metrics import MetricAverager, calc_metrics
from emr_configuration import (
    architecture_id as width_architecture_id, checkpoint_name,
    validate_hidden_channels, variant_suffix,
)


EMR_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPECTED_MSI_CHANNELS = {
    "PaviaU": 4,
    "Houston13": 8,
    "Chikusei": 8,
    "CAVE": 3,
    "Botswana": 8,
    "Augsburg": 4,
}

EMR_BACKBONE_WIDTH = 64
EMR_ARCHITECTURE_ID = "fixed64_dynamic_io_v2"


def capture_rng_state():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": (numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None,
    }


def restore_rng_state(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch_cpu"].cpu())
    if state.get("torch_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([value.cpu() for value in state["torch_cuda"]])


def _sample_or_resize(x, target_hw):
    target_h, target_w = int(target_hw[0]), int(target_hw[1])
    h, w = x.shape[-2:]
    if (h, w) == (target_h, target_w):
        return x
    if target_h <= h and target_w <= w and h % target_h == 0 and w % target_w == 0:
        step_h = h // target_h
        step_w = w // target_w
        return x[..., ::step_h, ::step_w][..., :target_h, :target_w]
    return F.interpolate(
        x,
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    dataset,
    degradation_mode,
    state_channels,
    filename=None,
    architecture_id=None,
    backbone_width=None,
    validation_metrics=None,
    best_metric=None,
    best_score=None,
    checkpoint_dir=None,
    best_epoch=None,
    bad_evals=0,
    training_protocol=None,
):
    checkpoint_dir = checkpoint_dir or os.path.join(
        EMR_ROOT, "checkpoints", degradation_mode, dataset
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    backbone_width = validate_hidden_channels(
        backbone_width if backbone_width is not None else getattr(model, "model_channels", EMR_BACKBONE_WIDTH)
    )
    architecture_id = architecture_id or width_architecture_id(backbone_width)
    if filename is None or filename == "last.pth.tar":
        filename = checkpoint_name("last", backbone_width)
    elif filename == "best.pth.tar":
        filename = checkpoint_name("best", backbone_width)
    model_out_path = os.path.join(checkpoint_dir, filename)

    payload = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "dataset": dataset,
        "degradation_mode": degradation_mode,
        "state_channels": int(state_channels),
        "architecture_id": str(architecture_id),
        "backbone_width": int(backbone_width),
        "training_stage": "stage1_registered",
        "optimizer_name": optimizer.__class__.__name__.lower(),
        "best_epoch": best_epoch,
        "bad_evals": int(bad_evals),
        "rng_state": capture_rng_state(),
    }
    if training_protocol is not None:
        payload["training_protocol"] = training_protocol
    if validation_metrics is not None:
        payload["validation_metrics"] = {
            key: float(value) for key, value in validation_metrics.items()
        }
    if best_metric is not None:
        payload["best_metric"] = str(best_metric)
    if best_score is not None:
        payload["best_score"] = float(best_score)

    temporary_path = model_out_path + ".tmp"
    torch.save(payload, temporary_path)
    os.replace(temporary_path, model_out_path)
    return model_out_path


def latest_checkpoint(dataset, degradation_mode, hidden_channels=EMR_BACKBONE_WIDTH):
    checkpoint_dir = os.path.join(
        EMR_ROOT, "checkpoints", degradation_mode, dataset
    )
    last_path = os.path.join(checkpoint_dir, checkpoint_name("last", hidden_channels))
    if os.path.isfile(last_path):
        return last_path
    if validate_hidden_channels(hidden_channels) != EMR_BACKBONE_WIDTH:
        raise FileNotFoundError(last_path)
    paths = glob.glob(os.path.join(checkpoint_dir, "model_epoch_*.pth.tar"))
    if not paths:
        raise FileNotFoundError(
            f"No EMR-Diff checkpoint found for {dataset}/{degradation_mode} "
            f"in {checkpoint_dir}"
        )

    def _epoch(path):
        stem = os.path.basename(path)
        return int(stem.split("model_epoch_")[-1].split(".pth")[0])

    return max(paths, key=_epoch)


def default_checkpoint(dataset, degradation_mode, hidden_channels=EMR_BACKBONE_WIDTH):
    best_path = os.path.join(
        EMR_ROOT,
        "checkpoints",
        degradation_mode,
        dataset,
        checkpoint_name("best", hidden_channels),
    )
    if os.path.exists(best_path):
        return best_path
    return latest_checkpoint(dataset, degradation_mode, hidden_channels)


class ResShiftTrainer:
    def __init__(self, configs):
        self.configs = configs
        self.epochs = int(self.configs.train["epochs"])
        self.num_timesteps = int(self.configs.diffusion.params.get("steps"))
        self.diffusion_sf = int(self.configs.diffusion.params.get("sf"))

        self.early_stop_metric = str(
            self.configs.train.get("early_stop_metric", "PSNR")
        )
        self.early_stop_patience = int(
            self.configs.train.get("early_stop_patience", 2)
        )
        self.early_stop_min_delta = float(
            self.configs.train.get("early_stop_min_delta", 0.02)
        )
        self.eval_seed = int(self.configs.train.get("eval_seed", 1234))
        if self.early_stop_metric != "PSNR":
            raise ValueError(
                "The shared comparison protocol currently requires PSNR for early stopping."
            )
        if self.early_stop_patience < 0:
            raise ValueError("early_stop_patience must be >= 0")
        if self.early_stop_min_delta < 0:
            raise ValueError("early_stop_min_delta must be >= 0")

        seed = int(self.configs.train.get("seed", 10))
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        requested_device = str(self.configs.train.get("device", "cuda"))
        if requested_device.startswith("cuda") and not torch.cuda.is_available():
            requested_device = "cpu"
        self.device = torch.device(requested_device)

        (
            self.train_dataloader,
            self.validation_dataloader,
            self.test_dataloader,
            self.data_info,
            self.shared_cfg,
        ) = build_ufg_loaders(self.configs)

        self.dataset = str(self.shared_cfg.dataset)
        requested_mode = str(self.configs.data.get("degradation_mode", "gaussian_bicubic"))
        adapter_mode = str(self.shared_cfg.degradation_mode)
        resolved_mode = str(self.data_info["degradation_mode"])
        if not (requested_mode == adapter_mode == resolved_mode):
            raise RuntimeError(
                "Degradation protocol mismatch: "
                f"requested={requested_mode}, adapter={adapter_mode}, resolved={resolved_mode}. "
                "Training is aborted before any checkpoint is written."
            )
        self.degradation_mode = resolved_mode

        self.hsi_channels = int(self.data_info["n_bands"])
        self.msi_channels = int(self.data_info["n_select_bands"])
        self.state_channels = self.hsi_channels + self.msi_channels

        expected_msi_channels = EXPECTED_MSI_CHANNELS.get(self.dataset)
        if expected_msi_channels is None:
            raise ValueError(f"Unsupported comparison dataset: {self.dataset}")
        if self.msi_channels != expected_msi_channels:
            raise ValueError(
                f"{self.dataset} comparison protocol requires {expected_msi_channels} "
                f"MSI channels, got {self.msi_channels}."
            )

        print(
            f"[EMR-Diff] dataset={self.dataset}, HSI={self.hsi_channels}, "
            f"MSI={self.msi_channels}, state={self.state_channels}, "
            f"scale=x{self.diffusion_sf}, degradation={self.degradation_mode}, "
            f"sensor={self.data_info.get('srf_profile')}"
        )
        print(
            f"[split] validation_rect={self.data_info.get('validation_rect')}, "
            f"test_rect={self.data_info.get('test_rect')}"
        )
        print(
            f"[early-stop] metric={self.early_stop_metric}, "
            f"min_delta={self.early_stop_min_delta}, "
            f"patience={self.early_stop_patience}, eval_seed={self.eval_seed}"
        )

        self._apply_dynamic_channel_config()
        self.build_model()
        self.build_diffusion_model()
        print(
            f"[EMR-Diff architecture] dataset={self.dataset} "
            f"state_channels={self.state_channels} "
            f"backbone_width={self.backbone_width} "
            f"architecture_id={self.architecture_id}"
        )
        self.edge_detector = Edge().to(self.device)
        self.setup_optimization()

        self.checkpoint_dir = str(
            self.configs.train.get(
                "checkpoint_dir",
                os.path.join(
                    EMR_ROOT, "checkpoints", self.degradation_mode, self.dataset
                ),
            )
        )
        self.output_dir = str(
            self.configs.train.get(
                "output_dir",
                os.path.join(
                    EMR_ROOT, "outputs", self.degradation_mode, self.dataset
                ),
            )
        )
        self.log_dir = str(
            self.configs.train.get(
                "log_dir",
                os.path.join(
                    EMR_ROOT, "logs", self.degradation_mode, self.dataset
                ),
            )
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.log_dir, exist_ok=True)

    def _apply_dynamic_channel_config(self):
        params = self.configs.model.params
        params.in_channels = self.state_channels
        params.model_channels = validate_hidden_channels(self.configs.train.get("hidden_channels", EMR_BACKBONE_WIDTH))
        params.out_channels = self.state_channels
        params.lqrgb_channels = self.state_channels
        params.rgb_channels = self.msi_channels
        self.configs.diffusion.params.band_dim = self.hsi_channels

    def setup_optimization(self):
        optimizer_name = str(
            self.configs.train.get("optimizer", "Adam")
        ).lower()
        lr = float(self.configs.train.get("lr"))
        weight_decay = float(self.configs.train.get("weight_decay", 0.0))
        if optimizer_name == "adamw":
            self.optimizer = torch.optim.AdamW(
                self.Net.parameters(), lr=lr, weight_decay=weight_decay
            )
        elif optimizer_name == "adam":
            self.optimizer = torch.optim.Adam(
                self.Net.parameters(), lr=lr, weight_decay=weight_decay
            )
        else:
            raise ValueError(
                f"Unsupported EMR-Diff optimizer={optimizer_name!r}; "
                "use Adam or AdamW."
            )
        self.optimizer_name = optimizer_name
        self.weight_decay = weight_decay

    def build_model(self):
        params = dict(self.configs.model.params)
        # Only internal features use the chosen width. State input, residual output,
        # conditions, losses and all diffusion steps retain HSI+MSI channels.
        self.Net = BAFUNet(**params).to(self.device)
        self.backbone_width = validate_hidden_channels(params["model_channels"])
        self.architecture_id = width_architecture_id(self.backbone_width)

    def build_diffusion_model(self):
        diffusion_opt = self.configs.get("diffusion", dict)
        self.EMRDIFF = EMRDIFF(diffusion_opt).to(self.device)

    def verify_checkpoint_architecture(self, checkpoint, *, context="checkpoint"):
        architecture_id = checkpoint.get("architecture_id")
        backbone_width = checkpoint.get("backbone_width")
        if architecture_id != self.architecture_id or int(backbone_width or -1) != self.backbone_width:
            raise ValueError(
                f"{context} uses incompatible EMR-Diff architecture: "
                f"architecture_id={architecture_id!r}, backbone_width={backbone_width}. "
                f"Current model requires {self.architecture_id!r} "
                f"with backbone_width={self.backbone_width}. "
                "Old hidden=state_channels and external state-projection "
                "checkpoints (fixed34/fixed64 v1) must be retrained."
            )
        if int(checkpoint.get("state_channels", -1)) != self.state_channels:
            raise ValueError(
                f"{context} state_channels={checkpoint.get('state_channels')} "
                f"but current dataset requires {self.state_channels}."
            )

    def _prepare_batch(self, batch):
        gt = batch["gt"].to(self.device, dtype=torch.float32, non_blocking=True)
        lq = batch["lr_hsi"].to(self.device, dtype=torch.float32, non_blocking=True)
        msi = batch["hr_msi"].to(self.device, dtype=torch.float32, non_blocking=True)
        return gt, lq, msi

    def _pseudo_msi(self, gt):
        if gt.shape[1] < self.msi_channels:
            raise ValueError(
                f"HSI has {gt.shape[1]} bands, cannot form {self.msi_channels}-band pseudo-MSI."
            )
        return gt[:, : self.msi_channels, :, :]

    def _condition(self, lq, msi, target_hw):
        lq_scaled = _sample_or_resize(lq, target_hw)
        msi_scaled = _sample_or_resize(msi, target_hw)
        return torch.cat((lq_scaled, msi_scaled), dim=1)

    def _x_start(self, gt):
        return torch.cat((gt, self._pseudo_msi(gt)), dim=1)

    def _multiscale_loss(self, network_output, up_out, x_start, lq, msi):
        loss_func = nn.L1Loss()
        hr_condition = self._condition(lq, msi, x_start.shape[-2:])
        loss = loss_func(network_output + hr_condition, x_start)
        for index in (2, 4, 6):
            if index >= len(up_out):
                continue
            feature = up_out[index]
            target_hw = feature.shape[-2:]
            target = _sample_or_resize(x_start, target_hw)
            condition = self._condition(lq, msi, target_hw)
            if feature.shape != target.shape or condition.shape != target.shape:
                raise RuntimeError(
                    f"Multi-scale shape mismatch at up_out[{index}]: "
                    f"feature={tuple(feature.shape)}, condition={tuple(condition.shape)}, "
                    f"target={tuple(target.shape)}"
                )
            loss = loss + loss_func(feature + condition, target)
        return loss

    def _reconstruct(self, lq, msi):
        hr_hw = msi.shape[-2:]
        lq_hr = _sample_or_resize(lq, hr_hw)
        condition = torch.cat((lq_hr, msi), dim=1)
        rgb_edge = self.edge_detector(msi)
        indices = list(range(self.num_timesteps))[::-1]
        noise = torch.randn_like(condition)
        x_t = self.EMRDIFF.prior_sample(condition, noise, edge_map=rgb_edge)

        self.Net.eval()
        with torch.no_grad():
            for t in indices:
                tt = torch.tensor(
                    [t] * x_t.shape[0], device=x_t.device, dtype=torch.long
                )
                x_pred, _ = self.Net(x_t, msi, lq_hr, tt)
                x_pred = x_pred + condition
                noise = torch.randn_like(x_pred)
                x_t = self.EMRDIFF.inverse_denoise(
                    x_start=x_pred,
                    x_t=x_t,
                    t=tt,
                    noise=noise,
                    edge_map=rgb_edge,
                )
        return x_t[:, : self.hsi_channels, :, :]

    def evaluate(self, loader=None, save_predictions=False, split_name="validation"):
        loader = loader or self.validation_dataloader
        averager = MetricAverager()

        cuda_devices = []
        if self.device.type == "cuda":
            device_index = (
                self.device.index
                if self.device.index is not None
                else torch.cuda.current_device()
            )
            cuda_devices = [device_index]

        with torch.random.fork_rng(devices=cuda_devices):
            torch.manual_seed(self.eval_seed)
            if self.device.type == "cuda":
                torch.cuda.manual_seed_all(self.eval_seed)

            for step, batch in enumerate(
                tqdm(loader, desc=f"EMR-Diff {split_name} eval",
                     dynamic_ncols=True, mininterval=1.0, disable=None)
            ):
                gt, lq, msi = self._prepare_batch(batch)
                prediction = self._reconstruct(lq, msi)
                metric_values = calc_metrics(
                    prediction, gt, scale_ratio=self.diffusion_sf
                )
                averager.update(metric_values)
                tqdm.write(
                    " ".join(
                        f"{name}={value:.6f}"
                        for name, value in metric_values.items()
                    )
                )

                if save_predictions:
                    sio.savemat(
                        os.path.join(
                            self.output_dir,
                            f"prediction_{split_name}_{step}.mat",
                        ),
                        {
                            "data": prediction.squeeze(0).detach().cpu().numpy(),
                            "gt": gt.squeeze(0).detach().cpu().numpy(),
                        },
                    )

        average = averager.average()
        print(
            f"[EMR-Diff:{self.dataset}:{self.degradation_mode}:{split_name}] "
            + " ".join(f"{k}={v:.6f}" for k, v in average.items())
        )

        metrics_name = "metrics.txt" if split_name == "test" else "validation_metrics.txt"
        metrics_path = os.path.join(self.output_dir, metrics_name)
        with open(metrics_path, "w", encoding="utf-8") as f:
            f.write(f"split: {split_name}\n")
            f.write(f"dataset: {self.dataset}\n")
            f.write(f"degradation_mode: {self.degradation_mode}\n")
            f.write(f"srf_profile: {self.data_info.get('srf_profile')}\n")
            f.write(f"eval_seed: {self.eval_seed}\n")
            for key, value in average.items():
                f.write(f"{key}: {value:.8f}\n")
        return average

    def registered_training_protocol(self):
        return {
            "scale_ratio": self.diffusion_sf,
            "diffusion_steps": self.num_timesteps,
            **{key: self.configs.data.get(key) for key in (
                "patch_size", "stride", "validation_size", "test_size",
                "mtf_nyquist", "psf_truncate",
            )},
        }

    def resume_training(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        self.verify_checkpoint_architecture(checkpoint, context=f"stage-1 resume {checkpoint_path}")
        if checkpoint.get("training_stage") not in (None, "stage1_registered"):
            raise ValueError("Stage-1 resume requires a registered checkpoint, not stage-2 mixed weights")
        if checkpoint.get("dataset") != self.dataset:
            raise ValueError("Resume checkpoint dataset mismatch")
        if checkpoint.get("degradation_mode") != self.degradation_mode:
            raise ValueError("Resume checkpoint degradation_mode mismatch")
        optimizer_name = checkpoint.get("optimizer_name")
        if optimizer_name is not None and optimizer_name != self.optimizer_name:
            raise ValueError(f"Resume optimizer mismatch: {optimizer_name} vs {self.optimizer_name}")
        for key, value in checkpoint.get("training_protocol", {}).items():
            if value != self.registered_training_protocol().get(key):
                raise ValueError(f"Resume training protocol mismatch: {key}")
        if "optimizer_state_dict" not in checkpoint:
            raise ValueError("Resume checkpoint has no optimizer state; cannot perform full training resume")
        completed = int(checkpoint["epoch"])
        if completed < 0:
            raise ValueError("Completed stage-1 epoch count must be nonnegative")
        # Stage 1 saves completed epoch counts (800 means next epoch is 801).
        self.Net.load_state_dict(checkpoint["model_state_dict"], strict=True)
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.weight_decay = float(self.optimizer.param_groups[0].get("weight_decay", 0.0))
        self.start_epoch = completed
        self.resume_best_score = float(checkpoint.get("best_score", checkpoint.get("validation_metrics", {}).get(self.early_stop_metric, float("-inf"))))
        self.resume_best_epoch = checkpoint.get("best_epoch")
        self.resume_bad_evals = int(checkpoint.get("bad_evals", 0))
        if checkpoint.get("rng_state") is not None:
            restore_rng_state(checkpoint["rng_state"])
            rng_status = "restored"
        else:
            rng_status = "not recorded in this older checkpoint"
        print(
            f"[resume] checkpoint={checkpoint_path} completed_epochs={completed} "
            f"next_epoch={completed + 1} best_{self.early_stop_metric}={self.resume_best_score:.6f} "
            f"lr={self.optimizer.param_groups[0]['lr']:.3e} optimizer=restored RNG={rng_status}"
        )
        return completed

    def train(self, epoch, verbose):
        start_epoch = getattr(self, "start_epoch", 0)
        if int(epoch) <= start_epoch:
            raise ValueError(f"Total target epochs ({epoch}) must exceed completed epochs ({start_epoch})")
        suffix = variant_suffix(self.backbone_width)
        history_path = os.path.join(self.log_dir, f"train_loss{suffix}.csv")
        if not os.path.exists(history_path):
            with open(history_path, "w", encoding="utf-8") as f:
                f.write("epoch,loss\n")

        validation_history_path = os.path.join(
            self.log_dir, f"validation_history{suffix}.csv"
        )
        if not os.path.exists(validation_history_path):
            with open(validation_history_path, "w", encoding="utf-8") as f:
                f.write(
                    "epoch,PSNR,SAM,best_score,bad_evals,is_best\n"
                )

        best_score = getattr(self, "resume_best_score", float("-inf"))
        best_epoch = getattr(self, "resume_best_epoch", None)
        bad_evals = getattr(self, "resume_bad_evals", 0)

        def save_training_checkpoint(current_epoch, filename=None, validation_metrics=None):
            return save_checkpoint(
                self.Net, self.optimizer, current_epoch, self.dataset,
                self.degradation_mode, self.state_channels, filename=filename,
                validation_metrics=validation_metrics, best_metric=self.early_stop_metric,
                best_score=best_score, best_epoch=best_epoch, bad_evals=bad_evals,
                checkpoint_dir=self.checkpoint_dir,
                training_protocol=self.registered_training_protocol(),
                architecture_id=self.architecture_id, backbone_width=self.backbone_width,
            )

        for epoch_index in range(start_epoch, int(epoch)):
            current_epoch = epoch_index + 1
            self.Net.train()
            running_loss = 0.0
            num_batches = 0
            progress = tqdm(
                self.train_dataloader,
                desc=(
                    f"EMR-Diff {self.dataset} {self.degradation_mode} "
                    f"epoch {current_epoch}"
                ),
                dynamic_ncols=True, mininterval=1.0, disable=None,
            )

            for batch in progress:
                gt, lq, msi = self._prepare_batch(batch)
                x_start = self._x_start(gt)
                hr_condition = self._condition(lq, msi, gt.shape[-2:])
                tt = torch.randint(
                    0,
                    self.num_timesteps,
                    size=(gt.shape[0],),
                    device=self.device,
                )
                noise = torch.randn_like(hr_condition)

                self.optimizer.zero_grad(set_to_none=True)
                x_t = self.EMRDIFF.forward_addnoise(
                    x_start=x_start,
                    y=hr_condition,
                    t=tt,
                    noise=noise,
                    rgb_hr=msi,
                )
                lq_hr = _sample_or_resize(lq, gt.shape[-2:])
                network_output, up_out = self.Net(x_t, msi, lq_hr, tt)
                loss = self._multiscale_loss(
                    network_output, up_out, x_start, lq, msi
                )
                loss.backward()
                self.optimizer.step()

                running_loss += float(loss.item())
                num_batches += 1
                progress.set_postfix(loss=f"{loss.item():.6f}", refresh=False)

            mean_loss = running_loss / max(num_batches, 1)
            with open(history_path, "a", encoding="utf-8") as f:
                f.write(f"{current_epoch},{mean_loss:.8f}\n")
            print(
                f"[EMR-Diff:{self.dataset}:{self.degradation_mode}] "
                f"epoch={current_epoch} train_loss={mean_loss:.6f}"
            )

            if current_epoch % int(verbose) != 0 and current_epoch != int(epoch):
                save_training_checkpoint(current_epoch, filename=checkpoint_name("last", self.backbone_width))
                continue

            validation_metrics = self.evaluate(
                loader=self.validation_dataloader,
                save_predictions=False,
                split_name="validation",
            )
            if self.early_stop_metric not in validation_metrics:
                raise KeyError(
                    f"Early-stop metric {self.early_stop_metric!r} not found in "
                    f"validation metrics: {sorted(validation_metrics.keys())}"
                )

            current_score = float(validation_metrics[self.early_stop_metric])
            improved = current_score > best_score + self.early_stop_min_delta

            if improved:
                best_score = current_score
                best_epoch = current_epoch
                bad_evals = 0
                best_path = save_training_checkpoint(
                    current_epoch,
                    filename=checkpoint_name("best", self.backbone_width),
                    validation_metrics=validation_metrics,
                )
                print(
                    f"[best] epoch={best_epoch} {self.early_stop_metric}="
                    f"{best_score:.6f} checkpoint={best_path}"
                )
            else:
                bad_evals += 1
                print(
                    f"[early-stop] no significant improvement: current "
                    f"{self.early_stop_metric}={current_score:.6f}, "
                    f"best={best_score:.6f} at epoch={best_epoch}, "
                    f"bad_evals={bad_evals}/{self.early_stop_patience}"
                )

            last_path = save_training_checkpoint(
                current_epoch, filename=checkpoint_name("last", self.backbone_width), validation_metrics=validation_metrics,
            )
            print(f"checkpoint={last_path}")

            with open(validation_history_path, "a", encoding="utf-8") as f:
                f.write(
                    f"{current_epoch},"
                    f"{float(validation_metrics.get('PSNR', float('nan'))):.8f},"
                    f"{float(validation_metrics.get('SAM', float('nan'))):.8f},"
                    f"{best_score:.8f},{bad_evals},{int(improved)}\n"
                )

            if (
                self.early_stop_patience > 0
                and bad_evals >= self.early_stop_patience
            ):
                print(
                    f"[early-stop] triggered at epoch={current_epoch}; "
                    f"best_epoch={best_epoch}, best_{self.early_stop_metric}="
                    f"{best_score:.6f}. Use {checkpoint_name('best', self.backbone_width)} for final test."
                )
                break

    def load_checkpoint(self, checkpoint_path=None):
        checkpoint_path = checkpoint_path or default_checkpoint(
            self.dataset, self.degradation_mode, self.backbone_width
        )
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.verify_checkpoint_architecture(
            checkpoint, context=str(checkpoint_path)
        )
        checkpoint_mode = checkpoint.get("degradation_mode")
        if checkpoint_mode is not None and checkpoint_mode != self.degradation_mode:
            raise ValueError(
                f"Checkpoint degradation_mode={checkpoint_mode} does not match "
                f"requested mode={self.degradation_mode}."
            )
        if "model_state_dict" in checkpoint:
            self.Net.load_state_dict(checkpoint["model_state_dict"])
        elif "model" in checkpoint:
            self.Net.load_state_dict(checkpoint["model"].state_dict())
        else:
            raise KeyError(f"Unsupported checkpoint format: {checkpoint_path}")
        print(f"loaded checkpoint={checkpoint_path}")
        return checkpoint_path

    def test(self, checkpoint_path=None):
        self.load_checkpoint(checkpoint_path)
        parameter_count = sum(p.numel() for p in self.Net.parameters())
        print(f"Params: {parameter_count}")
        return self.evaluate(
            loader=self.test_dataloader,
            save_predictions=True,
            split_name="test",
        )
