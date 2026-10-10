"""Batch evaluation of UAFL registered-only and HSI-deformed-mixed checkpoints.

Four synthetic x4 datasets. For EACH dataset:
  - registered-trained checkpoint: Registered test only
  - mixed HSI-deformed-trained checkpoint: Registered AND Warp tests

The original-scale Augsburg-2 Wald x3 task uses a separate test protocol.
No checkpoint is trained or modified by this script.
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

DATASETS = ("PaviaU", "Houston13", "Botswana", "Augsburg")
TEST = Path(__file__).resolve().parent / "test_hsi_deformed.py"
ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "dataset", "trained_condition", "tested_condition", "checkpoint",
    "checkpoint_epoch", "test_samples", "cases_per_patch",
    "PSNR", "SSIM", "ERGAS", "SAM", "CC", "RMSE", "RMSE_x255",
)


def parse_args():
    p = argparse.ArgumentParser(description="Four-dataset UAFL final 3-way x4 synthetic test")
    p.add_argument("--datasets", nargs="+", choices=DATASETS, default=list(DATASETS))
    p.add_argument("--data_root", default="./data/raw")
    p.add_argument("--registered_root", default="comparison/UAFL/checkpoints/physical")
    p.add_argument("--mixed_root", default="comparison/UAFL/checkpoints/hsi_warp_final")
    p.add_argument("--output_dir", default="comparison/UAFL/outputs/final_four_datasets")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--cases", type=int, default=10)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--dry_run", action="store_true", help="Show jobs and checkpoint paths without GPU")
    return p.parse_args()


def jobs(args):
    for dataset in args.datasets:
        yield dataset, "registered", "registered_only", (
            Path(args.registered_root) / dataset / "best.pth.tar"
        )
        yield dataset, "hsi_deformed_mixed", "registered_and_warp", (
            Path(args.mixed_root) / dataset / "best.pth.tar"
        )


def main():
    args = parse_args()
    if args.cases < 1:
        raise ValueError("Nonregistered testing needs cases >= 1")
    all_jobs = list(jobs(args))
    missing = [str(p) for _, _, _, p in all_jobs if not p.is_file()]
    for dataset, train_condition, test_mode, ckpt in all_jobs:
        print(f"PLAN {dataset} train={train_condition} test={test_mode} checkpoint={ckpt}", flush=True)
    if args.dry_run:
        if missing:
            print("NOT_PRESENT_CHECKPOINTS:")
            print("\n".join(missing))
        return
    if missing:
        raise FileNotFoundError(
            "Missing one or more checkpoints; no tests started. "
            "Override --registered_root/--mixed_root if needed:\n" + "\n".join(missing)
        )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for job_num, (dataset, trained_condition, test_mode, checkpoint) in enumerate(all_jobs, 1):
        json_file = output / f"{dataset}_{trained_condition}_seed{args.seed}.json"
        cmd = [
            sys.executable, str(TEST),
            "--dataset", dataset,
            "--data_root", args.data_root,
            "--image_size", "128", "--patch_size", "64", "--stride", "32",
            "--scale_ratio", "4", "--degradation_mode", "physical",
            "--mtf_nyquist", "0.2", "--psf_truncate", "3.0",
            "--msi_mode", "srf", "--srf_band_set", "auto",
            "--max_translation", "4.0", "--max_rotation_deg", "2.0",
            "--max_local_px", "4.0", "--control_grid", "5",
            "--min_jacobian", "0.5",
            "--seed", str(args.seed), "--cases", str(args.cases),
            "--device", args.device,
            "--test_mode", test_mode,
            "--checkpoint", str(checkpoint),
            "--output_json", str(json_file),
        ]
        print(f"RUN {job_num}/{len(all_jobs)} {dataset} {trained_condition}", flush=True)
        subprocess.run(cmd, cwd=ROOT, check=True)
        payload = json.loads(json_file.read_text(encoding="utf-8"))
        metrics = payload["average_metrics"]
        expected = ("Registered",) if test_mode == "registered_only" else ("Registered", "Warp")
        if tuple(metrics) != expected:
            raise RuntimeError(f"Unexpected test groups {tuple(metrics)} for {json_file}")
        conditions = payload["test_conditions"]
        for tested_condition, row in metrics.items():
            record = {
                "dataset":dataset, "trained_condition":trained_condition,
                "tested_condition":tested_condition,
                "checkpoint":str(checkpoint),
                "checkpoint_epoch":conditions.get("checkpoint_epoch"),
                "test_samples":conditions.get("test_samples"),
                "cases_per_patch":(
                    conditions["cases_per_patch"] if tested_condition=="Warp" else 0
                ),
            }
            record.update({key:row[key] for key in ("PSNR","SSIM","ERGAS","SAM","CC","RMSE","RMSE_x255")})
            rows.append(record)

    csv_file = output / f"UAFL_final_four_datasets_seed{args.seed}.csv"
    with csv_file.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print("\nFINAL_UAFL_FOUR_DATASETS_TABLE")
    print("Dataset    Trained            Tested      PSNR      SSIM     ERGAS       SAM        CC      RMSE*255")
    for r in rows:
        print(f"{r['dataset']:<10} {r['trained_condition']:<18} {r['tested_condition']:<10} "
              f"{r['PSNR']:>8.4f} {r['SSIM']:>9.6f} {r['ERGAS']:>9.4f} "
              f"{r['SAM']:>9.4f} {r['CC']:>9.6f} {r['RMSE_x255']:>11.4f}")
    print(f"SUMMARY_CSV={csv_file.resolve()}")
    print("EVALUATION_ONLY: no checkpoint was updated")


if __name__ == "__main__":
    main()
