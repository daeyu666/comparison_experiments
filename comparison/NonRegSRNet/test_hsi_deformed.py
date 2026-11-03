"""NonRegSRNet: independent self-supervised fits using the shared test observations."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import sys
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

import torch
from comparison.NonRegSRNet.common import (
    HERE,add_data_args,add_geometry_args,add_fit_args,build_shared_cfg,fit_config,
    geometry_kwargs,resolve_device,require_srf_weights,
    make_deformed_lr_hsi,sample_synthetic_geometry,
)
from comparison.NonRegSRNet.fit import PairwisePredictor
from comparison.NonRegSRNet.model import ARCHITECTURE_ID,UPSTREAM_COMMIT
from data_loader import build_datasets
from metrics import calc_metrics,MetricAverager


def parser(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p);add_geometry_args(p);add_fit_args(p)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--eval_seed",type=int,default=1234)
    p.add_argument("--cases",type=int,default=10)
    p.add_argument("--checkpoint",default="",help="No pretrained checkpoint is used by this pairwise method")
    p.add_argument("--test_mode",choices=["registered_only","registered_and_warp"],default="registered_and_warp")
    p.add_argument("--output_json",default="")
    p.add_argument("--max_samples",type=int,default=0,help="Smoke-test only; 0 evaluates all heldout patches")
    return p.parse_args(argv)


def build_test_components(args):
    if getattr(args,"checkpoint",""):
        raise ValueError("NonRegSRNet fits each observation pair from scratch; use --nonreg_cache_dir for pair caches")
    device=resolve_device(args.device)
    _,_,test_set,info=build_datasets(build_shared_cfg(args),include_validation=True)
    if not len(test_set):
        raise ValueError("NonRegSRNet test split is empty")
    srf=require_srf_weights(info).float().cpu()
    config=fit_config(args)
    predictor=PairwisePredictor(srf,args.scale_ratio,config,
                               cache_dir=getattr(args,"nonreg_cache_dir","") or None)
    p0=test_set.degradation_operator.to(device)
    details=dict(architecture_id=ARCHITECTURE_ID,upstream_commit=UPSTREAM_COMMIT,
                 training_stage="observation_pair_self_supervision",
                 inference_protocol="fresh fit per LR-HSI/HR-MSI pair; fixed iteration budget",
                 fit_config=asdict(config),gt_used_for_optimization=False,
                 hsi_channels=int(info["n_bands"]),msi_channels=int(info["n_select_bands"]),
                 sensor_fingerprint=hashlib.sha256(srf.contiguous().numpy().tobytes()).hexdigest())
    return test_set,predictor,p0,device,details


def main(argv=None):
    args=parser(argv)
    if args.cases<1 or args.max_samples<0:
        raise ValueError("cases must be positive; max_samples must be nonnegative")
    test_set,predictor,p0,device,details=build_test_components(args)
    registered,warped=MetricAverager(),MetricAverager()
    rows=[]
    count=len(test_set) if not args.max_samples else min(args.max_samples,len(test_set))
    for index in range(count):
        batch=test_set[index]
        gt=batch["gt"][None].to(device,dtype=torch.float32)
        msi=batch["hr_msi"][None].to(device,dtype=torch.float32)
        # GT is used only to synthesize the benchmark observation and score the result.
        # The optimizer receives exclusively lr_hsi and hr_msi.
        with torch.no_grad():
            lr=p0.degrade(gt)
        pred=predictor(lr,msi)
        reg=calc_metrics(pred,gt,args.scale_ratio)
        registered.update(reg)
        if args.test_mode=="registered_and_warp":
            generator=torch.Generator(device=device).manual_seed(args.seed+70000)
            for case in range(args.cases):
                with torch.no_grad():
                    geometry=sample_synthetic_geometry(
                        *gt.shape[-2:],device=device,dtype=gt.dtype,generator=generator,
                        local_strength_min_fraction=0.,local_strength_max_fraction=1.,
                        **geometry_kwargs(args))
                    lr_warp=make_deformed_lr_hsi(gt,geometry,p0)
                pred=predictor(lr_warp,msi)
                scores=calc_metrics(pred,gt,args.scale_ratio)
                warped.update(scores)
                rows.append(dict(sample=index+1,case=case+1,metrics=scores,
                                 observation_sha256=predictor.last_report["observation_sha256"]))
        print(f"NonRegSRNet pair {index+1}/{count}: Registered PSNR={reg['PSNR']:.6f}",flush=True)
    result=dict(method="NonRegSRNet",**details,test_conditions=vars(args),
                test_samples=count,smoke_test=bool(args.max_samples),
                average_metrics={"Registered":registered.average(),"Warp":warped.average() or None},
                per_sample_case=rows)
    output=Path(args.output_json) if args.output_json else HERE/"outputs"/"pairwise"/args.dataset/(args.test_mode+".json")
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps(result["average_metrics"],indent=2),flush=True)
    print(f"FINAL_TEST_JSON={output}",flush=True)
    return result


if __name__=="__main__":
    main()
