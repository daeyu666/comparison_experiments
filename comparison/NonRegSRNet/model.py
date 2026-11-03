"""Released NonRegSRNet observation-only objective and component assembly.

This is an image-pair optimizer: the final MSI abundance decoder is fitted jointly
with the observed LR-HSI branch. A frozen amortized MSI-only forward is NOT the
paper's nonregistered inference protocol.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import upstream_core as core

UPSTREAM_COMMIT = "84e236db23695120ce0eb2a2f25b78d7fde70c21"
UPSTREAM_NETWORK_BLOB = "6f23cef4a6371b6d99d58087b2df4e58097a6d73"
ARCHITECTURE_ID = "nonregsrnet_released_blind_pairwise_v1"


@dataclass(frozen=True)
class FitConfig:
    # Official bash.sh demo: 2000 constant + 8000 decaying iterations.
    iterations: int = 10000
    constant_steps: int = 2000
    decay_steps: int = 8000
    lr: float = 3e-4
    num_rho: int = 100
    alpha: float = 1.0
    beta: float = 0.01
    gamma: float = 0.01
    delta: float = 1.0
    theta: float = 0.01
    ncc_window: int = 1
    seed: int = 1234

    def validate(self):
        if min(self.iterations, self.num_rho, self.decay_steps, self.ncc_window) < 1:
            raise ValueError("iterations, num_rho, decay_steps and ncc_window must be positive")
        if self.constant_steps < 0 or self.ncc_window % 2 != 1 or self.lr <= 0:
            raise ValueError("Invalid schedule, LR or NCC window")
        if min(self.alpha, self.beta, self.gamma, self.delta, self.theta) < 0:
            raise ValueError("Loss coefficients must be nonnegative")
        return self


class NonRegSRNet(nn.Module):
    def __init__(self, hsi_channels, msi_channels, srf_weights, scale_ratio=4, num_rho=100):
        super().__init__()
        self.hsi_channels = int(hsi_channels)
        self.msi_channels = int(msi_channels)
        self.scale_ratio = int(scale_ratio)
        self.num_rho = int(num_rho)
        if min(self.hsi_channels, self.msi_channels, self.scale_ratio, self.num_rho) < 1:
            raise ValueError("Channel counts, scale and num_rho must be positive")
        weights = torch.as_tensor(srf_weights, dtype=torch.float32).detach().clone()
        if weights.shape != (self.msi_channels, self.hsi_channels):
            raise ValueError("Require shared SRF with shape [MSI, HSI]")
        if not torch.isfinite(weights).all() or (weights < 0).any():
            raise ValueError("SRF weights must be finite and nonnegative")
        ranges = []
        for row in weights:
            index = (row > 0).nonzero(as_tuple=True)[0]
            if not index.numel():
                raise ValueError("Each MSI channel needs nonzero SRF support")
            ranges.append((int(index[0]), int(index[-1])))
        self.register_buffer("observation_srf", weights)
        self.net_MSI2A = core.Msi2A(self.msi_channels, self.num_rho, ngf=64, useClamp=True)
        self.net_A2Img = core.A2Img(self.num_rho, self.hsi_channels)
        self.net_HSI2A = core.HSI2A(self.hsi_channels, self.num_rho, ngf=64, useClamp=True)
        self.net_PSF = core.PSF(self.scale_ratio)
        self.net_HSI2MSI = core.ConvolutionHsi2Msi(
            self.hsi_channels, self.msi_channels,
            torch.tensor(ranges).cpu().numpy(),
        )
        self.net_Displacefiled = core.UNet(2*self.num_rho, 2)
        self.net_Spacetransform = core.SpatialTransformation(use_gpu=False)
        for module in (self.net_MSI2A, self.net_A2Img, self.net_HSI2A, self.net_Displacefiled):
            core.init_weights(module, "kaiming")
        core.init_weights(self.net_PSF, "mean_space")
        core.init_weights(self.net_HSI2MSI, "mean_channel")
        self.sum_to_one = core.SumToOneLoss()

    def reconstruct(self, hr_msi):
        return self.net_A2Img(self.net_MSI2A(hr_msi))

    def forward(self, lr_hsi, hr_msi, return_aux=False):
        if lr_hsi.ndim != 4 or hr_msi.ndim != 4 or lr_hsi.shape[0] != hr_msi.shape[0]:
            raise ValueError("Require paired BCHW observations")
        if lr_hsi.shape[1] != self.hsi_channels or hr_msi.shape[1] != self.msi_channels:
            raise ValueError("NonRegSRNet observation channel mismatch")
        if tuple(s*self.scale_ratio for s in lr_hsi.shape[-2:]) != hr_msi.shape[-2:]:
            raise ValueError("NonRegSRNet observation spatial ratio mismatch")
        if min(lr_hsi.shape[-2:]) < 2:
            raise ValueError("LR-HSI must be at least 2x2 for flow smoothness/BatchNorm")
        lr_a = self.net_HSI2A(lr_hsi)
        lr_rec = self.net_A2Img(lr_a)
        hr_a = self.net_MSI2A(hr_msi)
        fused = self.net_A2Img(hr_a)
        msi_rec = self.net_HSI2MSI(fused)
        msi_lr_a = self.net_PSF(hr_a)
        flow = self.net_Displacefiled(torch.cat((lr_a, msi_lr_a), dim=1)).permute(0,2,3,1)
        # Direction matches release: HSI abundance -> reliable MSI coordinate system.
        recovered_a = self.net_Spacetransform(lr_a.permute(0,2,3,1), flow).permute(0,3,1,2)
        if not return_aux:
            return fused
        return dict(fused=fused, lr_a=lr_a, hr_a=hr_a, lr_rec=lr_rec, msi_rec=msi_rec,
                    msi_lr_a=msi_lr_a, recovered_a=recovered_a, flow=flow)

    def observation_loss(self, lr_hsi, hr_msi, config):
        """No HR-HSI argument: GT cannot enter any optimization term."""
        out = self(lr_hsi, hr_msi, return_aux=True)
        terms = {
            "lr_pixelwise": F.l1_loss(out["lr_rec"], lr_hsi, reduction="sum"),
            "lr_sumtoone": self.sum_to_one(out["lr_a"]) * config.beta,
            "msi_pixelwise": F.l1_loss(out["msi_rec"], hr_msi, reduction="sum") * config.alpha,
            "msi_sumtoone": self.sum_to_one(out["hr_a"]) * config.beta,
            "recovered_pixelwise": F.l1_loss(
                self.net_A2Img(out["recovered_a"]), self.net_PSF(out["fused"]),
                reduction="sum") * config.delta,
        }
        ncc = core.cross_correlation_loss(out["recovered_a"], out["msi_lr_a"], config.ncc_window)
        smooth = core.smooothing_loss(out["flow"].permute(0,3,1,2))
        terms["voxmorph"] = (1-ncc)*config.gamma + smooth*config.theta
        total = sum(terms.values())
        return total, {**terms, "ncc": ncc, "smooth": smooth}

    def make_optimizer(self, config):
        displacement = list(self.net_Displacefiled.parameters())
        rest = [p for name,p in self.named_parameters() if not name.startswith("net_Displacefiled.")]
        # One Adam with parameter groups gives the same independent tensor updates
        # as the six official Adam optimizers; flow uses half the base LR.
        return torch.optim.Adam([
            {"params":rest,"lr":config.lr},
            {"params":displacement,"lr":config.lr*0.5},
        ], betas=(0.9,0.999), weight_decay=0.0)

    @torch.no_grad()
    def project_parameters(self):
        clip = core.ZeroOneClipper()
        self.net_A2Img.apply(clip)
        self.net_PSF.apply(clip)
        self.net_HSI2MSI.apply(core.SumToOneClipper())


def model_metadata(model, config):
    return dict(method="NonRegSRNet", architecture_id=ARCHITECTURE_ID,
                upstream_commit=UPSTREAM_COMMIT, upstream_network_blob=UPSTREAM_NETWORK_BLOB,
                inference_protocol="fresh self-supervised fit per observation pair",
                hsi_channels=model.hsi_channels, msi_channels=model.msi_channels,
                scale_ratio=model.scale_ratio, num_rho=model.num_rho,
                base_encoder_width=64, fit_config=asdict(config),
                parameters=sum(p.numel() for p in model.parameters()),
                sensors="blind learned scale-stride PSF and support-constrained SRF")
