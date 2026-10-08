"""Full-resolution Augsburg-2 Wald inference for strictly Wald-trained UAFL.

Input: observed 30m HSI + REAL Sentinel-2 10m MSI, sub_area_2.
Output: x3 -> 10m 242-band fused HSI, without 10m HSI labels or PSNR.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

THIS=Path(__file__).resolve().parent
sys.path.insert(0,str(THIS))
from augsburg2_wald_common import (
    correct_msi, load_state, read_json, read_radiometry, require_wald, predict_uafl
)
from model import build_uafl


def parse_args():
    p=argparse.ArgumentParser(description="Strict Augsburg-2 Wald UAFL 10m inference")
    p.add_argument("--wald_root",default="./data/augsburg2_wald")
    p.add_argument("--checkpoint",default="./comparison/UAFL/checkpoints/augsburg2_wald/best.pth.tar")
    p.add_argument("--radiometry_json",default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--save_root",default="./comparison/UAFL/outputs/augsburg2_wald")
    p.add_argument("--tile_size",type=int,default=96)
    p.add_argument("--tile_stride",type=int,default=48)
    p.add_argument("--device",default="cuda")
    p.add_argument("--write_tif",action="store_true")
    return p.parse_args()


def positions(length, tile, stride):
    if length<tile:
        raise ValueError("Full scene dimension is smaller than tile")
    out=list(range(0,length-tile+1,stride))
    last=length-tile
    if not out or out[-1]!=last:
        out.append(last)
    if any(x%3 for x in out):
        raise ValueError("Tiling origins must be divisible by 3 for the LR grid")
    return out


@torch.no_grad()
def main():
    args=parse_args()
    if args.tile_size%24 or args.tile_stride%24 or args.tile_stride>args.tile_size:
        raise ValueError("UAFL full tiles/strides must be multiples of 24; stride<=tile")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    sigma=require_wald(args.wald_root)
    calibration=read_radiometry(args.radiometry_json)
    sha=hashlib.sha256(Path(args.radiometry_json).read_bytes()).hexdigest()
    device=torch.device(args.device)

    model=build_uafl(4).to(device)
    ckpt=load_state(args.checkpoint,model,device=device)
    if ckpt.get("radiometry_sha256")!=sha or abs(float(ckpt.get("wald_sigma",-1))-sigma)>1e-8:
        raise ValueError("Wald inference checkpoint radiometry/PSF mismatch")
    model.eval()

    root=Path(args.wald_root)/"full"
    meta=read_json(root/"meta.json")
    if meta.get("region")!="sub_area_2":
        raise ValueError("Expected exact sub_area_2 Wald original-resolution inputs")
    lr=np.load(root/"lr_hsi.npy",mmap_mode="r")
    msi=np.load(root/"hr_msi.npy",mmap_mode="r")
    mask=np.load(root/"valid_mask.npy",mmap_mode="r")
    h,w,c=msi.shape
    if (c!=4 or lr.shape!=(h//3,w//3,242) or
        mask.shape!=(h,w) or h%6 or w%6):
        raise ValueError(f"Bad full Wald array shapes HSI={lr.shape}, MSI={msi.shape}, mask={mask.shape}")

    ys=positions(h,args.tile_size,args.tile_stride)
    xs=positions(w,args.tile_size,args.tile_stride)
    sum_cube=np.zeros((h,w,242),dtype=np.float32)
    count=np.zeros((h,w),dtype=np.float32)

    for i,top in enumerate(ys):
        for j,left in enumerate(xs):
            ts=args.tile_size
            lp=np.asarray(lr[top//3:(top+ts)//3,left//3:(left+ts)//3]).copy()
            mp=np.asarray(msi[top:top+ts,left:left+ts]).copy()
            x_h=torch.from_numpy(np.ascontiguousarray(lp.transpose(2,0,1))).float().unsqueeze(0).to(device)
            x_m=torch.from_numpy(np.ascontiguousarray(mp.transpose(2,0,1))).float().unsqueeze(0).to(device)
            x_m=correct_msi(x_m,calibration)
            pred=predict_uafl(model,x_h,x_m)
            arr=pred[0].permute(1,2,0).float().cpu().numpy()
            sum_cube[top:top+ts,left:left+ts]+=arr
            count[top:top+ts,left:left+ts]+=1.0
            print(f"UAFL_FULL_TILE {i+1}/{len(ys)} {j+1}/{len(xs)} at={top},{left}")
    if not np.all(count>0):
        raise RuntimeError("Full-resolution reconstruction has uncovered pixels")
    fused=sum_cube/count[...,None]
    fused[np.asarray(mask)==0]=0.
    dest=Path(args.save_root)
    dest.mkdir(parents=True,exist_ok=True)
    output=dest/"Augsburg2_Wald_UAFL_full_HSI.npy"
    np.save(output,fused.astype(np.float32))

    if args.write_tif:
        try:
            import rasterio
            from affine import Affine
        except ImportError as exc:
            raise ImportError("GeoTIFF output requires rasterio and affine") from exc
        geotiff=dest/"Augsburg2_Wald_UAFL_full_HSI.tif"
        with rasterio.open(
            geotiff,"w",driver="GTiff",height=h,width=w,count=242,
            dtype="float32",crs=meta["crs"],
            transform=Affine(*meta["transform_6"]),compress="deflate",tiled=True,
        ) as dst:
            for k in range(242):
                dst.write(fused[:,:,k],k+1)
        print(f"UAFL_WALD_GEOTIFF={geotiff}")

    report={
        "protocol":"Augsburg-2-Wald-UAFL",
        "inputs":"observed 30m HSI and original real Sentinel-2 10m MSI",
        "region":"sub_area_2",
        "full_reference_HSI_10m":False,
        "quantitative_full_psnr":None,
        "output_shape":list(fused.shape),
        "valid_fraction":float(np.asarray(mask).mean()),
        "tile_size":args.tile_size,
        "tile_stride":args.tile_stride,
        "checkpoint":str(Path(args.checkpoint).resolve()),
        "srf_source":"fixed Wald cache",
        "radiometry_sha256":sha,
    }
    (dest/"UAFL_Wald_full_protocol.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2),encoding="utf-8"
    )
    print(f"UAFL_WALD_FULL_OUTPUT={output} shape={fused.shape} HR_HSI_REFERENCE=unavailable")


if __name__=="__main__":
    main()
