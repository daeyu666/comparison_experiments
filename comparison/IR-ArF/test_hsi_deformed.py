"""Evaluate IR&ArF on the shared registered and HSI-warped test protocol."""
from __future__ import annotations
import argparse, json, random, sys, hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
if str(HERE) not in sys.path: sys.path.insert(0,str(HERE))

from common import add_data_args,add_geometry_args,build_shared_cfg,make_generator,resolve_device,set_seed,require_srf_weights
from data_loader import build_datasets
from comparison.CLSNet.common import make_deformed_lr_hsi,sample_synthetic_geometry
from comparison.CLSNet.final_metrics import average_metrics,calc_metrics,format_metrics
from model import Main, UPSTREAM_COMMIT, IRARF_ARCHITECTURE_ID


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--checkpoint",default="")
    p.add_argument("--test_mode",choices=["registered_only","registered_and_warp"],default="registered_and_warp")
    p.add_argument("--cases",type=int,default=10)
    p.add_argument("--geometry_seeds",type=int,nargs="+",default=list(range(10001,10011)))
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--eval_seed",type=int,default=1234)
    p.add_argument("--output_json",default="")
    return p.parse_args()


def build_test_components(args, *, expected_stage):
    """Load one IR&ArF checkpoint with dataset/architecture/protocol checks."""
    if expected_stage not in ("registered", "mixed"):
        raise ValueError(f"Unknown IR&ArF stage: {expected_stage}")
    if (args.image_size, args.patch_size, args.stride, args.scale_ratio)!=(128,64,32,4):
        raise ValueError("IR&ArF shared benchmark requires image/patch/stride 128/64/32 and x4")
    device=resolve_device(args.device)
    _,_,test_set,info=build_datasets(build_shared_cfg(args),include_validation=True)
    p0=test_set.degradation_operator.to(device)
    model_args=SimpleNamespace(bands=int(info["n_bands"]), msi_channels=int(info["n_select_bands"]),
                               gpu=(device.index or 0) if device.type=="cuda" else 0)
    model=Main(bands=model_args.bands,args=model_args).to(device)
    family="physical" if expected_stage=="registered" else "hsi_warp_final"
    ckpt=Path(args.checkpoint) if getattr(args,"checkpoint","") else HERE/"checkpoints"/family/args.dataset/"best.pth.tar"
    state=torch.load(ckpt,map_location=device,weights_only=False)
    if state.get("method")!="IR-ArF" or state.get("stage")!=expected_stage:
        raise ValueError(f"IR&ArF checkpoint method/stage mismatch: {ckpt}")
    if state.get("architecture_id")!=IRARF_ARCHITECTURE_ID:
        raise ValueError("IR&ArF checkpoint architecture version mismatch")
    if state.get("upstream_commit")!=UPSTREAM_COMMIT:
        raise ValueError("IR&ArF checkpoint model-source version mismatch")
    expected_srf=hashlib.sha256(
        torch.as_tensor(require_srf_weights(info),dtype=torch.float32).cpu().numpy().tobytes()
    ).hexdigest()
    if state.get("sensor_fingerprint")!=expected_srf:
        raise ValueError("IR&ArF checkpoint SRF fingerprint mismatch")
    saved_model_args=state.get("model_args",{})
    for name in ("bands","msi_channels"):
        if int(saved_model_args.get(name,-1))!=int(getattr(model_args,name)):
            raise ValueError(f"IR&ArF checkpoint sensor mismatch: {name}")
    trained=state.get("args")
    if not isinstance(trained,dict):
        raise ValueError("IR&ArF checkpoint is missing training protocol")
    for name in ("dataset","image_size","patch_size","stride","scale_ratio",
                 "degradation_mode","mtf_nyquist","psf_truncate",
                 "degradation_sigma","degradation_kernel_size","msi_mode",
                 "srf_path","wavelength_path","wavelength_root","srf_interp","srf_band_set"):
        if hasattr(args,name) and name in trained and trained[name]!=getattr(args,name):
            raise ValueError(f"IR&ArF checkpoint protocol mismatch: {name}")
    if expected_stage=="mixed":
        for name in ("max_translation","max_rotation_deg","max_local_px",
                     "control_grid","min_jacobian","registered_probability"):
            if hasattr(args,name) and name in trained and trained[name]!=getattr(args,name):
                raise ValueError(f"IR&ArF checkpoint geometry mismatch: {name}")
    model.load_state_dict(state["model"],strict=True)
    model.eval()
    return test_set,model,p0,device,state


def main():
    args=parse_args()
    if args.image_size!=128 or args.patch_size!=64 or args.stride!=32 or args.scale_ratio!=4:
        raise ValueError("The shared benchmark protocol is fixed at image/patch/stride=128/64/32 and x4")
    if args.cases<1: raise ValueError("--cases must be positive")
    set_seed(args.seed)
    stage="registered" if args.test_mode=="registered_only" else "mixed"
    test_set,model,p0,device,state=build_test_components(args,expected_stage=stage)
    ckpt=Path(args.checkpoint) if args.checkpoint else HERE/"checkpoints"/(
        "physical" if stage=="registered" else "hsi_warp_final"
    )/args.dataset/"best.pth.tar"
    reg_rows,warp_rows,per_case=[],[],[]
    warp_by_seed={seed:[] for seed in args.geometry_seeds}
    with torch.no_grad():
        for idx in range(len(test_set)):
            item=test_set[idx]
            gt=item["gt"].unsqueeze(0).to(device,dtype=torch.float32)
            msi=item["hr_msi"].unsqueeze(0).to(device,dtype=torch.float32)
            lr=p0.degrade(gt)
            pred=model(msi,lr,gt.shape[-1])[0]
            reg_rows.append(calc_metrics(pred,gt,args.scale_ratio))
            if args.test_mode=="registered_and_warp":
                for geometry_seed in args.geometry_seeds:
                    geom_gen=make_generator(device,geometry_seed)
                    for case in range(args.cases):
                        geometry=sample_synthetic_geometry(
                            gt.shape[-2],gt.shape[-1],device=device,dtype=gt.dtype,generator=geom_gen,
                            max_translation=args.max_translation,max_rotation_deg=args.max_rotation_deg,
                            max_local_px=args.max_local_px,control_grid=args.control_grid,
                            min_jacobian=args.min_jacobian,local_strength_min_fraction=0.0,
                            local_strength_max_fraction=1.0,
                        )
                        warped=make_deformed_lr_hsi(gt,geometry,p0)
                        pred_w=model(msi,warped,gt.shape[-1])[0]
                        met=calc_metrics(pred_w,gt,args.scale_ratio)
                        warp_rows.append(met)
                        warp_by_seed[geometry_seed].append(met)
                        per_case.append({"sample":idx+1,"geometry_seed":geometry_seed,"case":case+1,
                            "dx_hr_px":float(geometry.dx.item()),"dy_hr_px":float(geometry.dy.item()),
                            "rotation_deg":float(geometry.theta_deg.item()),"metrics":met})
    reg=average_metrics(reg_rows)
    seed_means={seed:average_metrics(rows) for seed,rows in warp_by_seed.items() if rows}
    warp=average_metrics(list(seed_means.values())) if seed_means else None
    warp_std=({key:float(np.std([metrics[key] for metrics in seed_means.values()],ddof=1))
               for key in next(iter(seed_means.values())).keys()} if len(seed_means)>1 else None)
    print("IRARF_FINAL_TEST")
    print("REGISTERED",format_metrics(reg))
    if warp is not None:
        print("WARP_MEAN_OVER_GEOMETRY_SEEDS",format_metrics(warp))
        print("WARP_STD_OVER_GEOMETRY_SEEDS",warp_std)
    result={
        "method":"IR&ArF","dataset":args.dataset,"checkpoint":str(ckpt),
        "checkpoint_epoch":state.get("epoch"),"test_mode":args.test_mode,
        "test_patches":len(test_set),"registered_metrics":reg,"warp_metrics":warp,"warp_seed_std":warp_std,
        "warp_metrics_by_seed":seed_means,
        "cases_per_patch":args.cases if warp is not None else 0,
        "geometry_seeds":args.geometry_seeds if warp is not None else [],
        "geometry_seed":args.seed+70000,"eval_seed":args.eval_seed,
        "observation":"LR-HSI=P0(W_phi(X)); HR-MSI=R0(X); GT=X",
        "per_case":per_case,
    }
    out=Path(args.output_json) if args.output_json else HERE/"outputs"/args.dataset/f"IRArF_{args.test_mode}_seed{args.seed}.json"
    out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding="utf-8")
    print("FINAL_TEST_JSON="+str(out))


if __name__=="__main__":
    main()
