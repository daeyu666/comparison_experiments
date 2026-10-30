"""M2DTN final test under the shared synthetic x4 protocol."""
from __future__ import annotations
import argparse, json, random, sys
from pathlib import Path
import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from comparison.M2DTN.common import (
    HERE, add_data_args, add_geometry_args, build_shared_cfg, load_checkpoint,
    make_generator, model_for_info, resolve_device, make_deformed_lr_hsi,
    sample_synthetic_geometry,
)
from comparison.UAFL.hsi_deformation import jacobian_determinant
from data_loader import build_datasets
from metrics import calc_metrics


def parse_args(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p); add_geometry_args(p)
    p.add_argument("--seed",type=int,default=10)
    p.add_argument("--cases",type=int,default=10)
    p.add_argument("--eval_seed",type=int,default=1234)
    p.add_argument("--checkpoint",default="")
    p.add_argument("--test_mode",choices=["registered_only","registered_and_warp"],default="registered_and_warp")
    p.add_argument("--print_cases",action="store_true")
    p.add_argument("--output_json",default="")
    return p.parse_args(argv)


def _average(rows):
    if not rows: return None
    return {key:float(np.mean([row[key] for row in rows])) for key in rows[0]}


def main():
    args=parse_args()
    if args.test_mode=="registered_and_warp" and args.cases<1:
        raise ValueError("--cases must be >=1")
    if (args.image_size,args.patch_size,args.stride,args.scale_ratio)!=(128,64,32,4):
        raise ValueError("Unified test protocol is fixed at 128/64/32/x4")

    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(args.seed)
    device=resolve_device(args.device)
    _,_,test_set,info=build_datasets(build_shared_cfg(args),include_validation=True)
    if not len(test_set): raise ValueError("Empty M2DTN test split")
    p0=test_set.degradation_operator.to(device)
    model=model_for_info(info,args,device)
    stage="registered" if args.test_mode=="registered_only" else "mixed"
    family=args.degradation_mode if stage=="registered" else "hsi_warp_final"
    ckpt_path=Path(args.checkpoint) if args.checkpoint else HERE/"checkpoints"/family/args.dataset/"best.pth.tar"
    checkpoint=load_checkpoint(ckpt_path,model,args,expected_stage=stage)
    model.eval()
    print(f"M2DTN_TEST_CHECKPOINT stage={stage} epoch={checkpoint['epoch']} path={ckpt_path}")

    registered_rows=[]; warp_rows=[]; per_case=[]
    cuda_devices=[]
    if device.type=="cuda":
        cuda_devices=[device.index if device.index is not None else torch.cuda.current_device()]
    with torch.no_grad(),torch.random.fork_rng(devices=cuda_devices):
        torch.manual_seed(args.eval_seed)
        if cuda_devices: torch.cuda.manual_seed_all(args.eval_seed)
        for sample_idx in range(len(test_set)):
            item=test_set[sample_idx]
            gt=item["gt"].unsqueeze(0).to(device,dtype=torch.float32)
            msi=item["hr_msi"].unsqueeze(0).to(device,dtype=torch.float32)
            pred=model(p0.degrade(gt),msi)
            registered_rows.append(calc_metrics(pred,gt,args.scale_ratio))
            if args.test_mode=="registered_and_warp":
                generator=make_generator(device,args.seed+70000)
                for case_idx in range(args.cases):
                    geometry=sample_synthetic_geometry(
                        gt.shape[-2],gt.shape[-1],device=device,dtype=gt.dtype,
                        generator=generator,max_translation=args.max_translation,
                        max_rotation_deg=args.max_rotation_deg,max_local_px=args.max_local_px,
                        control_grid=args.control_grid,min_jacobian=args.min_jacobian,
                        local_strength_min_fraction=0.0,local_strength_max_fraction=1.0,
                    )
                    warped_lr=make_deformed_lr_hsi(gt,geometry,p0)
                    metrics=calc_metrics(model(warped_lr,msi),gt,args.scale_ratio)
                    warp_rows.append(metrics)
                    record=dict(
                        sample=sample_idx+1,case=case_idx+1,
                        dx_hr_px=float(geometry.dx.item()),dy_hr_px=float(geometry.dy.item()),
                        rotation_deg=float(geometry.theta_deg.item()),
                        local_max_hr_px=float(torch.linalg.vector_norm(geometry.local_field,dim=1).amax().item()),
                        min_jacobian=float(jacobian_determinant(geometry.local_field).amin().item()),
                        metrics=metrics,
                    )
                    per_case.append(record)
                    if args.print_cases:
                        print(f"SAMPLE={sample_idx+1:04d} CASE={case_idx+1:02d} "
                              f"PSNR={metrics['PSNR']:.4f} SAM={metrics['SAM']:.4f}")

    result=dict(
        method="M2DTN",dataset=args.dataset,mode=stage,checkpoint=str(ckpt_path),
        checkpoint_epoch=int(checkpoint["epoch"]),registered_metrics=_average(registered_rows),
        warp_metrics=_average(warp_rows),per_case=per_case,
    )
    output=Path(args.output_json) if args.output_json else (
        HERE/"outputs"/"unified_two_stage"/args.dataset/f"M2DTN_{stage}_seed{args.seed}.json"
    )
    output.parent.mkdir(parents=True,exist_ok=True)
    output.write_text(json.dumps(result,indent=2),encoding="utf-8")
    print(f"M2DTN_RESULT registered={result['registered_metrics']} warp={result['warp_metrics']}")


if __name__=="__main__":
    main()
