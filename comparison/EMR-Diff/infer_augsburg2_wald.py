"""EMR-Diff native 10m inference for Augsburg-2 center-heldout Wald x3.

Only the untrained central 144x144 Sentinel-2 ROI is reconstructed and scored.
There is no 10m HSI ground truth; QNR/Dlambda/Ds use original observations.
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
    build_model, build_diffusion, read_json,
    read_radiometry, require_center_holdout, verify_checkpoint, predict,
)
from EMRDiff import Edge
from emr_configuration import checkpoint_name, validate_hidden_channels
from augsburg2_wald_qnr import evaluate_cache
from augsburg2_wald_center_roi import crop_heldout, geotiff_transform
from comparison.real_augsburg.common import predict_native_roi
from experiment_protocol import REAL_NATIVE_INFERENCE


def parse_args():
    p = argparse.ArgumentParser(
        description="EMR-Diff Augsburg-2 center-heldout native x3 inference"
    )
    p.add_argument(
        "--center_holdout", action="store_true",
        help="Explicit marker for the new central spatial holdout protocol",
    )
    p.add_argument(
        "--wald_root",
        default="../S2Diff-MH/data/augsburg2_wald_center_holdout",
    )
    p.add_argument(
        "--radiometry_json",
        default="../S2Diff-MH/data/calibration/Augsburg2_Wald_center_holdout_radiometry.json",
    )
    p.add_argument(
        "--checkpoint",
        default=None,
    )
    p.add_argument(
        "--save_root",
        default="./comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout",
    )
    p.add_argument("--device", default="cuda")
    p.add_argument("--write_tif", action="store_true")
    p.add_argument("--skip_qnr", action="store_true")
    p.add_argument("--qnr_window_hr", type=int, default=48)
    p.add_argument("--qnr_min_valid_fraction", type=float, default=0.8)
    p.add_argument("--qnr_support_fraction", type=float, default=0.01)
    p.add_argument("--eval_seed", type=int, default=1234)
    p.add_argument("--model_width", "--base_channels", type=validate_hidden_channels)
    args = p.parse_args()
    if args.checkpoint is None:
        args.checkpoint = str(Path("./comparison/EMR-Diff/checkpoints/augsburg2_wald_center_holdout")
                              / checkpoint_name("best", args.model_width))
    if args.skip_qnr:
        p.error("Center-heldout inference requires QNR output; omit --skip_qnr")
    return args


@torch.no_grad()
def predict_roi(model, diffusion, edge, lq, ref):
    """Keep any network padding outside the reconstructed observation grid."""
    h, w = ref.shape[-2:]
    lq_hr = F.interpolate(lq, size=(h, w), mode="bicubic", align_corners=False)
    ph, pw = (-h) % 16, (-w) % 16
    if ph or pw:
        pad = (0, pw, 0, ph)
        lq_hr = F.pad(lq_hr, pad, mode="replicate")
        ref = F.pad(ref, pad, mode="replicate")
    pred = predict(model, diffusion, edge, lq_hr, ref)
    return pred[..., :h, :w]


@torch.no_grad()
def main():
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    sigma, split_protocol_id, test_bbox_30m, forbidden_bbox_30m = (
        require_center_holdout(args.wald_root)
    )
    radiometry_path = Path(args.radiometry_json)
    calibration = read_radiometry(radiometry_path, args.wald_root)
    sha = hashlib.sha256(radiometry_path.read_bytes()).hexdigest()
    device = torch.device(args.device)

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    verify_checkpoint(
        checkpoint,
        radiometry_sha=sha,
        sigma=sigma,
        split_protocol_id=split_protocol_id,
        test_bbox_30m=test_bbox_30m,
        forbidden_bbox_30m=forbidden_bbox_30m,
        width=args.model_width,
    )
    model = build_model(int(checkpoint["model_width"]), device=device)
    model.load_state_dict(checkpoint.pop("model_state_dict"))
    checkpoint.pop("optimizer_state_dict", None)
    model.eval()
    diffusion = build_diffusion(device)
    edge = Edge().to(device).eval()

    root = Path(args.wald_root) / "full"
    meta = read_json(root / "meta.json")
    if meta.get("region") != "sub_area_2":
        raise ValueError("Augsburg-2 native inference requires sub_area_2")
    lr = np.load(root / "lr_hsi.npy", mmap_mode="r")
    msi = np.load(root / "hr_msi.npy", mmap_mode="r")
    mask = np.load(root / "valid_mask.npy", mmap_mode="r")
    h, w, channels = msi.shape
    if (
        channels != 4
        or lr.shape != (h // 3, w // 3, 242)
        or mask.shape != (h, w)
        or h % 6
        or w % 6
    ):
        raise ValueError(
            f"Invalid original Wald arrays: HSI={lr.shape}, "
            f"MSI={msi.shape}, mask={mask.shape}"
        )

    lr, msi, mask, roi_y0, roi_x0, suffix, selected_protocol = crop_heldout(
        args.wald_root,
        lr,
        msi,
        mask,
        ckpt_protocol_id=checkpoint.get("split_protocol_id"),
        ckpt_bbox_30m=checkpoint.get("test_bbox_30m"),
    )
    if suffix != "heldout" or selected_protocol != split_protocol_id:
        raise ValueError("EMR-Diff center protocol may reconstruct heldout ROI only")
    h, w = msi.shape[:2]
    if (h, w) != (144, 144):
        raise ValueError(f"Expected 144x144 native heldout ROI, got {(h,w)}")
    print(
        f"EMR_WALD_INFERENCE spatial_protocol={selected_protocol} "
        f"area={suffix} HR_shape={h}x{w} "
        f"origin_10m_rowcol=({roi_y0},{roi_x0})"
    )

    torch.manual_seed(args.eval_seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.eval_seed)

    fused = predict_native_roi(lambda hsi, msi: predict_roi(model, diffusion, edge, hsi, msi),
                               lr, msi, calibration, device)
    fused[np.asarray(mask) == 0] = 0.0

    dest = Path(args.save_root)
    dest.mkdir(parents=True, exist_ok=True)
    output = dest / "Augsburg2_Wald_EMRDiff_heldout_HSI.npy"
    np.save(output, fused.astype(np.float32))

    if args.write_tif:
        try:
            import rasterio
        except ImportError as exc:
            raise ImportError("GeoTIFF requires rasterio and affine") from exc
        tif = dest / "Augsburg2_Wald_EMRDiff_heldout_HSI.tif"
        with rasterio.open(
            tif,
            "w",
            driver="GTiff",
            height=h,
            width=w,
            count=242,
            dtype="float32",
            crs=meta["crs"],
            transform=geotiff_transform(
                meta["transform_6"], roi_y0, roi_x0
            ),
            compress="deflate",
            tiled=True,
        ) as writer:
            for band in range(242):
                writer.write(fused[:, :, band], band + 1)
        print(f"EMR_WALD_GEOTIFF={tif}")

    quality = evaluate_cache(
        args.wald_root,
        str(output),
        args.radiometry_json,
        window_hr=args.qnr_window_hr,
        min_valid_fraction=args.qnr_min_valid_fraction,
        support_fraction=args.qnr_support_fraction,
    )
    qpath = dest / "EMRDiff_Wald_heldout_QNR.json"
    qpath.write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"EMR_WALD_HSI_MSI_QNR QNR={quality['QNR']:.6f} "
        f"Dlambda={quality['Dlambda']:.6f} Ds={quality['Ds']:.6f} "
        "SPECTRAL_REFERENCE=242_band_LR_HSI "
        "SPATIAL_REFERENCE=original_HR_MSI"
    )

    report = {
        "protocol": "Augsburg-2-Wald-EMR-Diff",
        "split_protocol_id": split_protocol_id,
        "evaluation_area": "heldout",
        "test_bbox_30m": test_bbox_30m,
        "inputs": "observed 30m HSI and original real Sentinel-2 10m MSI",
        "region": "sub_area_2",
        "full_reference_HSI_10m": False,
        "quantitative_full_psnr": None,
        "no_reference_quality": quality,
        "output_shape": list(fused.shape),
        "valid_fraction": float(np.asarray(mask).mean()),
        "inference_mode": REAL_NATIVE_INFERENCE,
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "radiometry_sha256": sha,
        "diffusion_eval_seed": args.eval_seed,
        "backbone_width": model.width,
        "checkpoint_epoch": checkpoint.get("epoch"),
    }
    protocol_path = dest / "EMRDiff_Wald_heldout_protocol.json"
    protocol_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        f"EMR_WALD_HELDOUT_OUTPUT={output} shape={fused.shape} "
        "HR_HSI_REFERENCE=unavailable"
    )


if __name__ == "__main__":
    main()
