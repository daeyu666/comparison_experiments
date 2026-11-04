"""Registered CNMF with author updates, known shared SRF and physical operator."""
from dataclasses import dataclass,asdict
import numpy as np
from . import upstream_core as core
from comparison.traditional.operators import validate_pair

SOURCE_ARCHIVE_SHA256="29acccea7c2109a9a7bd2ec9fd64f7aa9e0611fb073e043e7afe6e5f58eab55f"


@dataclass(frozen=True)
class CNMFConfig:
    iterations:int=200
    endmembers:int=0  # Released max(min(30,bands), VD at false-alarm .05).
    seed:int=1234


def fuse(lr_hsi,hr_msi,srf,operator,config=None):
    cfg=config or CNMFConfig()
    if cfg.iterations<2 or cfg.endmembers<0:
        raise ValueError("CNMF requires iterations>=2 and endmembers>=0")
    lr,msi,r=validate_pair(lr_hsi,hr_msi,srf,operator)
    channels,h,w=lr.shape;H,W=msi.shape[-2:]
    # Match the author's HWC -> pixels-by-bands -> transpose layout. With
    # overcomplete VCA, tiny BLAS/layout differences can select other vertices.
    hsi_hwc=lr.transpose(1,2,0).copy();msi_hwc=msi.transpose(1,2,0).copy()
    y=hsi_hwc.reshape(-1,channels).T;m=msi_hwc.reshape(-1,msi.shape[0]).T
    if not np.any(y):
        if np.any(m):
            raise ValueError("Zero LR-HSI has no identifiable spectral dictionary for nonzero MSI")
        return np.zeros((channels,H,W)),dict(config=asdict(cfg),endmembers=0)
    rank=cfg.endmembers or max(min(30,channels),int(np.real(core.vd(y,.05))))
    rank=min(rank,channels,y.shape[1])
    if rank<2:
        raise ValueError("CNMF/VCA needs at least two endmembers")
    delta=2*np.sqrt(msi_hwc.mean()/.7455)/msi.shape[0]**3
    rng=np.random.get_state()
    try:
        np.random.seed(cfg.seed)
        hyper,multi,wh,hh,wm,hm,_,_=core.CNMF_init(
            H,W,operator.scale,rank,y,m,delta,cfg.iterations,
            1e-8,1e-8,r,init_mode=0)
        # The released top-level default has one outer update. It returns the
        # updated HS dictionary with the initial high-resolution abundances.
        wh,*_=core.CNMF_ite(H,W,operator.scale,rank,hyper,multi,wh,hh,wm,hm,
            cfg.iterations,1e-8,1e-8,1,0,r,
            degrade_fn=lambda hwc:operator.forward(hwc.transpose(2,0,1)).transpose(1,2,0))
        out=(wh[:channels]@hm).reshape(channels,H,W)
    finally:
        np.random.set_state(rng)
    if not np.isfinite(out).all():
        raise FloatingPointError("CNMF produced nonfinite output")
    return out,dict(config=asdict(cfg),endmembers=rank,outer_iterations=1,
                    source_archive_sha256=SOURCE_ARCHIVE_SHA256,sensors="known shared SRF/P0")

