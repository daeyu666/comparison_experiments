"""EMR-Diff original-resolution Augsburg-2 Wald inference and HSI-MSI QNR.

Input: observed EnMAP-like 30m HSI and REAL, unwarped Sentinel-2 10m MSI.
Never evaluate original-resolution PSNR/SAM without a 10m HSI label.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from wald_emr_common import (
    build_model, build_diffusion, correct_msi, read_json,
    read_radiometry, require_wald, verify_checkpoint,
)
from EMRDiff import Edge
from augsburg2_wald_qnr import evaluate_cache


def parse_args():
    p = argparse.ArgumentParser(description="EMR-Diff strict Wald Augsburg2 full x3")
    p.add_argument("--wald_root", default="./data/augsburg2_wald")
    p.add_argument("--radiometry_json",
                   default="./data/calibration/Augsburg2_Wald_radiometry.json")
    p.add_argument("--checkpoint",
                   default="./comparison/EMR-Diff/checkpoints/augsburg2_wald/best.pth.tar")
    p.add_argument("--save_root",
                   default="./comparison/EMR-Diff/outputs/augsburg2_wald")
    p.add_argument("--tile_size", type=int, default=96)
    p.add_argument("--tile_stride", type=int, default=48)
    p.add_argument("--device", default="cuda")
    p.add_argument("--write_tif", action="store_true")
    p.add_argument("--skip_qnr", action="store_true")
    p.add_argument("--qnr_window_hr", type=int, default=48)
    p.add_argument("--qnr_min_valid_fraction", type=float, default=0.8)
    p.add_argument("--qnr_support_fraction", type=float, default=0.01)
    p.add_argument("--eval_seed", type=int, default=1234)
    return p.parse_args()


def positions(length, tile, stride):
    if length < tile:
        raise ValueError("Full scene dimension is smaller than a tile")
    out = list(range(0, length - tile + 1, stride))
    last = length - tile
    if not out or out[-1] != last:
        out.append(last)
    if any(x % 3 for x in out):
        raise ValueError("Full-resolution tile origins must respect x3 grid")
    return out


@torch.no_grad()
def main():
    args = parse_args()
    if (args.tile_size % 48 or args.tile_stride % 48
            or args.tile_stride > args.tile_size):
        raise ValueError("EMR full tiles and strides must be divisible by 48")
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    sigma = require_wald(args.wald_root)
    radiometry_path = Path(args.radiometry_json)
    calibration = read_radiometry(radiometry_path)
    sha = hashlib.sha256(radiometry_path.read_bytes()).hexdigest()
    device = torch.device(args.device)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    verify_checkpoint(checkpoint, radiometry_sha=sha, sigma=sigma)
    model = build_model(int(checkpoint["model_width"]), device=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    diffusion = build_diffusion(device)
    edge = Edge().to(device).eval()

    root = Path(args.wald_root) / "full"
    meta = read_json(root / "meta.json")
    if meta.get("region") != "sub_area_2":
        raise ValueError("Original Wald inference must use sub_area_2")
    lr = np.load(root/"lr_hsi.npy", mmap_mode="r")
    msi = np.load(root/"hr_msi.npy", mmap_mode="r")
    mask = np.load(root/"valid_mask.npy", mmap_mode="r")
    h, w, channels = msi.shape
    if (channels != 4 or lr.shape != (h//3, w//3, 242)
            or mask.shape != (h,w) or h%6 or w%6):
        raise ValueError(f"Invalid real Wald full arrays: {lr.shape}, {msi.shape}, {mask.shape}")

    ys = positions(h, args.tile_size, args.tile_stride)
    xs = positions(w, args.tile_size, args.tile_stride)
    sum_cube = np.zeros((h,w,242), dtype=np.float32)
    count = np.zeros((h,w), dtype=np.float32)
    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    for i, top in enumerate(ys):
        for j, left in enumerate(xs):
            sz = args.tile_size
            lp = np.asarray(lr[top//3:(top+sz)//3,
                               left//3:(left+sz)//3]).copy()
            mp = np.asarray(msi[top:top+sz,left:left+sz]).copy()
            lq = torch.from_numpy(np.ascontiguousarray(lp.transpose(2,0,1))).float()
            ref = torch.from_numpy(np.ascontiguousarray(mp.transpose(2,0,1))).float()
            lq = lq.unsqueeze(0).to(device)
            ref = correct_msi(ref.unsqueeze(0).to(device), calibration)
            lq_hr = F.interpolate(
                lq, size=(sz,sz), mode="bicubic", align_corners=False
            )
            condition = torch.cat((lq_hr,ref), dim=1)
            emap = edge(ref)
            state = diffusion.prior_sample(
                condition, torch.randn_like(condition), edge_map=emap
            )
            for step in range(diffusion.num_diffusion_timesteps - 1, -1, -1):
                t = torch.full((1,), step, device=device, dtype=torch.long)
                residual,_ = model(state, ref, lq_hr, t)
                start = residual + condition
                state = diffusion.inverse_denoise(
                    x_start=start, x_t=state, t=t,
                    noise=torch.randn_like(start), edge_map=emap,
                )
            pred = state[:,:242].squeeze(0).permute(1,2,0).float().cpu().numpy()
            sum_cube[top:top+sz,left:left+sz] += pred
            count[top:top+sz,left:left+sz] += 1.
            print(f"EMR_WALD_FULL_TILE {i+1}/{len(ys)} {j+1}/{len(xs)} at={top},{left}")
    if not np.all(count > 0):
        raise RuntimeError("Uncovered full reconstruction pixels")
    fused = sum_cube/count[...,None]
    fused[np.asarray(mask) == 0] = 0.
    dest = Path(args.save_root)
    dest.mkdir(parents=True, exist_ok=True)
    output = dest/"Augsburg2_Wald_EMRDiff_full_HSI.npy"
    np.save(output, fused.astype(np.float32))

    if args.write_tif:
        try:
            import rasterio
            from affine import Affine
        except ImportError as exc:
            raise ImportError("GeoTIFF requires rasterio and affine") from exc
        tif = dest/"Augsburg2_Wald_EMRDiff_full_HSI.tif"
        with rasterio.open(
            tif, "w", driver="GTiff", height=h, width=w, count=242,
            dtype="float32", crs=meta["crs"],
            transform=Affine(*meta["transform_6"]), compress="deflate", tiled=True,
        ) as writer:
            for k in range(242):
                writer.write(fused[:,:,k],k+1)
        print(f"EMR_WALD_GEOTIFF={tif}")

    quality = None
    if not args.skip_qnr:
        quality = evaluate_cache(
            args.wald_root, str(output), args.radiometry_json,
            window_hr=args.qnr_window_hr,
            min_valid_fraction=args.qnr_min_valid_fraction,
            support_fraction=args.qnr_support_fraction,
        )
        qpath = dest/"EMRDiff_Wald_full_QNR.json"
        qpath.write_text(json.dumps(quality,ensure_ascii=False,indent=2),
                         encoding="utf-8")
        print(
            f"EMR_WALD_HSI_MSI_QNR QNR={quality['QNR']:.6f} "
            f"Dlambda={quality['Dlambda']:.6f} Ds={quality['Ds']:.6f} "
            "SPECTRAL_REFERENCE=242_band_LR_HSI "
            "SPATIAL_REFERENCE=original_HR_MSI"
        )
    report = {
        "protocol": "Augsburg-2-Wald-EMR-Diff",
        "no_reference_quality": quality,
        "inputs": "observed 30m HSI and original real Sentinel-2 10m MSI",
        "region":"sub_area_2", "full_reference_HSI_10m":False,
        "quantitative_full_psnr":None, "output_shape":list(fused.shape),
        "valid_fraction":float(np.asarray(mask).mean()),
        "tile_size":args.tile_size, "tile_stride":args.tile_stride,
        "checkpoint":str(Path(args.checkpoint).resolve()),
        "radiometry_sha256":sha, "diffusion_eval_seed":args.eval_seed,
    }
    (dest/"EMRDiff_Wald_full_protocol.json").write_text(
        json.dumps(report,ensure_ascii=False,indent=2), encoding="utf-8"
    )
    print(f"EMR_WALD_FULL_OUTPUT={output} shape={fused.shape} HR_HSI_REFERENCE=unavailable")


if __name__ == "__main__":
    main()
