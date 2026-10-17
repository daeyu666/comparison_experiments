"""Rectangular DRTNet benchmark wrapper; source internals kept under upstream/.

Native model requires HR128/LR32. Reflect padding is used on HR64/LR16
training patches; the output is cropped back to the 64x64 supervised area.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.nn import functional as F
from comparison.DRTNet.upstream.models.MCT_rectangle import MCT_rectangle

UPSTREAM_COMMIT = "source-daeyu666-DRTNet-master-2026-10-17"
ARCHITECTURE_ID = "drtnet_rectangle_safa_cesr_benchmark_v1"

class DRTNet(nn.Module):
    def __init__(self, hsi_channels:int, msi_channels:int, scale_ratio:int=4,
                 dataset:str="PaviaU"):
        super().__init__()
        if scale_ratio != 4:
            raise ValueError("Native DRTNet supports only synthetic x4")
        self.hsi_channels=int(hsi_channels)
        self.msi_channels=int(msi_channels)
        self.scale_ratio=int(scale_ratio)
        self.dataset=str(dataset)
        self.core=MCT_rectangle("DRTnet",scale_ratio,msi_channels,hsi_channels,dataset=dataset)
        self.sk_group=self.core.sk_group

    def forward_with_aux(self, lr_hsi:torch.Tensor, hr_msi:torch.Tensor):
        if lr_hsi.ndim!=4 or hr_msi.ndim!=4 or lr_hsi.shape[0]!=hr_msi.shape[0]:
            raise ValueError("Expected matching BxCxHxW observations")
        b,c,lh,lw=lr_hsi.shape
        hh,hw=hr_msi.shape[-2:]
        if c!=self.hsi_channels or hr_msi.shape[1]!=self.msi_channels:
            raise ValueError("Observation band mismatch")
        if (hh,hw)!=(lh*4,lw*4):
            raise ValueError("DRTNet benchmark expects x4 grids")
        if hh!=hw or hh not in (64,128):
            raise ValueError("DRTNet benchmark supports square 64 or 128 HR patches")
        pad=(128-hh)//2
        if pad:
            lpad=pad//4
            hr_msi=F.pad(hr_msi,(pad,pad,pad,pad),mode="reflect")
            lr_hsi=F.pad(lr_hsi,(lpad,lpad,lpad,lpad),mode="reflect")
        output,aux=self.core(lr_hsi,hr_msi)
        if pad:
            output=output[...,pad:pad+hh,pad:pad+hw]
            aux=aux[...,pad:pad+hh,pad:pad+hw]
        if output.shape!=(b,self.hsi_channels,hh,hw):
            raise RuntimeError(f"Unexpected DRTNet output {tuple(output.shape)}")
        return output,aux

    def forward(self,lr_hsi,hr_msi):
        return self.forward_with_aux(lr_hsi,hr_msi)[0]

def build_drtnet(hsi_channels,msi_channels,scale_ratio,dataset=None):
    return DRTNet(hsi_channels,msi_channels,scale_ratio,dataset or "PaviaU")

def architecture_metadata(model):
    return dict(architecture_id=ARCHITECTURE_ID,
                source_commit=UPSTREAM_COMMIT,sk_group=model.sk_group,
                patch_adapter="reflect-pad HR64->128, LR16->32; center-crop",
                hsi_channels=model.hsi_channels,msi_channels=model.msi_channels,
                scale_ratio=model.scale_ratio)

def verify_architecture(checkpoint,model):
    for key,value in architecture_metadata(model).items():
        if checkpoint.get(key)!=value:
            raise ValueError(f"DRTNet architecture mismatch {key}: {checkpoint.get(key)!r} vs {value!r}")

def parameter_count(model):
    return sum(p.numel() for p in model.parameters())
