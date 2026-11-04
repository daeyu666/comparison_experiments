"""Shared Registered-only evaluator: observations -> numerical fusion -> public metrics."""
import argparse
import csv
import hashlib
import importlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from comparison.UAFL.common import build_shared_cfg,require_srf_weights
from data_loader import build_datasets
from metrics import calc_metrics,MetricAverager
from experiment_protocol import DATASETS
from .operators import PhysicalOperator
from .options import add_classical_args

ROOT=Path(__file__).resolve().parents[2]


def parser(method,argv=None):
    p=argparse.ArgumentParser(description=method+" registered-only classical fusion; no pretraining")
    p.add_argument("--dataset",choices=DATASETS,default="PaviaU")
    p.add_argument("--mode",choices=["registered"],default="registered")
    p.add_argument("--data_root",default="./data/raw")
    p.add_argument("--device",default="cpu",help="Shared interface; the numerical solver runs on CPU")
    p.add_argument("--output_dir",default="")
    p.add_argument("--max_samples",type=int,default=0)
    p.add_argument("--eval_seed",type=int,default=1234)
    add_classical_args(p)
    return p.parse_args(argv)


def run(method,args):
    if method not in ("CNMF","HySure") or args.mode!="registered":
        raise ValueError("CNMF and HySure support Registered-only evaluation")
    if args.max_samples<0 or args.classical_threads<1:
        raise ValueError("max_samples>=0 and classical_threads>=1 required")
    # These are the same frozen inputs as all supervised Registered baselines.
    args.image_size,args.patch_size,args.stride,args.scale_ratio=128,64,32,4
    args.degradation_mode,args.mtf_nyquist,args.psf_truncate="physical",.2,3.
    args.msi_mode,args.srf_interp,args.srf_band_set="srf","pchip","auto"
    args.srf_path=args.wavelength_path=""
    args.wavelength_root="./data/wavelengths"
    _,_,data,info=build_datasets(build_shared_cfg(args),include_validation=True)
    if not len(data):
        raise ValueError("Empty shared test split")
    srf=require_srf_weights(info).double().cpu().numpy()
    p0=data.degradation_operator.to("cpu")
    operator=PhysicalOperator(4,.2,3.)
    module=importlib.import_module("comparison."+method+".model")
    if method=="CNMF":
        cfg=module.CNMFConfig(args.cnmf_iterations,args.cnmf_endmembers,args.eval_seed)
    else:
        cfg=module.HySureConfig(iterations=args.hysure_iterations,rank=args.hysure_rank,
            lambda_phi=args.hysure_lambda_phi,lambda_m=args.hysure_lambda_m,
            cg_maxiter=args.hysure_cg_maxiter,cg_rtol=args.hysure_cg_rtol)
    output=Path(args.output_dir) if args.output_dir else ROOT/"comparison"/method/"outputs"/"registered"/args.dataset
    output.mkdir(parents=True,exist_ok=True)
    average=MetricAverager();rows=[];reports=[]
    count=min(args.max_samples,len(data)) if args.max_samples else len(data)
    with threadpool_limits(limits=args.classical_threads):
        for idx in range(count):
            item=data[idx]
            gt=item["gt"].detach().float().cpu()[None]
            msi=item["hr_msi"].detach().float().cpu()
            with torch.no_grad():
                lr=p0.degrade(gt)[0]
            start=time.perf_counter()
            prediction,report=module.fuse(lr.numpy(),msi.numpy(),srf,operator,cfg)
            elapsed=time.perf_counter()-start
            scores=calc_metrics(torch.from_numpy(prediction)[None],gt,4)
            average.update(scores)
            rows.append(dict(sample=idx+1,seconds=elapsed,**scores))
            reports.append(report)
            if args.classical_save_predictions:
                np.save(output/f"sample_{idx+1:04d}.npy",prediction.astype(np.float32))
            print(f"{method} Registered {idx+1}/{count}: PSNR={scores['PSNR']:.6f} SAM={scores['SAM']:.6f} seconds={elapsed:.3f}",flush=True)
    metrics=average.average()
    result=dict(method=method,mode="registered",protocol="shared-registered-physical-x4",
        sensor_mode="known shared SRF and spatial operator",actual_device="cpu",
        sensor_fingerprint=hashlib.sha256(srf.tobytes()).hexdigest(),
        test_samples=count,smoke_test=bool(args.max_samples),
        average_metrics={"Registered":metrics},per_sample=rows,solver_reports=reports,
        args=vars(args),gt_used_for_optimization=False,
        total_solver_seconds=sum(row["seconds"] for row in rows))
    (output/"summary.json").write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    for name,values in [("per_sample_metrics.csv",rows),("summary.csv",[dict(condition="Registered",**metrics)])]:
        with (output/name).open("w",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(values[0]))
            writer.writeheader();writer.writerows(values)
    print("FINAL_REGISTERED "+" ".join(f"{k}={v:.6f}" for k,v in metrics.items()),flush=True)
    print(f"FINAL_TEST_JSON={output/'summary.json'}",flush=True)
    return result


def main(method,argv=None):
    return run(method,parser(method,argv))

