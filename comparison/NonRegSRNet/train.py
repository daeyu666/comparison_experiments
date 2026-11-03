"""Fit released NonRegSRNet on one observed pair (no HR-HSI labels required)."""
import argparse
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from comparison.NonRegSRNet.common import add_fit_args,fit_config,resolve_device
from comparison.NonRegSRNet.fit import fit_pair


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pair",required=True,help="NPZ containing lr_hsi and hr_msi arrays only")
    p.add_argument("--srf_weights",required=True,help="Shared MSI-by-HSI SRF weights .npy")
    p.add_argument("--layout",choices=["hwc","chw"],default="hwc")
    p.add_argument("--scale_ratio",type=int,default=4)
    p.add_argument("--device",default="cuda:0")
    p.add_argument("--eval_seed",type=int,default=1234)
    p.add_argument("--output_dir",required=True)
    add_fit_args(p)
    args=p.parse_args(argv)
    device=resolve_device(args.device)
    pair=np.load(args.pair)
    def tensor(name):
        value=np.asarray(pair[name],dtype=np.float32)
        if value.ndim!=3:
            raise ValueError("Observed pair arrays must be HWC or CHW")
        if args.layout=="hwc":
            value=value.transpose(2,0,1)
        return torch.from_numpy(np.ascontiguousarray(value))[None].to(device)
    prediction,report=fit_pair(tensor("lr_hsi"),tensor("hr_msi"),
                              np.load(args.srf_weights),args.scale_ratio,fit_config(args),
                              cache_dir=args.nonreg_cache_dir or None,progress=True)
    out=Path(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    np.save(out/"fused_hsi.npy",prediction[0].permute(1,2,0).cpu().numpy())
    (out/"fit_protocol.json").write_text(json.dumps(report,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
