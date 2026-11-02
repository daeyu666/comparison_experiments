"""Train the official IR&ArF architecture on the shared HSI-MSI protocol."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))

from common import (  # noqa: E402
    add_data_args, add_geometry_args, build_shared_cfg, deformed_observation,
    make_generator, resolve_device, set_seed,
)
from data_loader import build_datasets  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from model import Main  # noqa: E402


class NCC:
    """Official local normalized cross-correlation loss, device-safe."""
    def __init__(self, win=None):
        self.win=win

    def loss(self, y_true, y_pred):
        I,J=y_true,y_pred
        ndims=I.ndim-2
        if ndims not in (1,2,3):
            raise ValueError(f"NCC expects 1D/2D/3D spatial tensors, got {ndims}")
        win=[9]*ndims if self.win is None else self.win
        sum_filt=torch.ones([1,1,*win],device=I.device,dtype=I.dtype)
        pad=math.floor(win[0]/2)
        if ndims==1:
            stride,padding=1,pad
        elif ndims==2:
            stride,padding=(1,1),(pad,pad)
        else:
            stride,padding=(1,1,1),(pad,pad,pad)
        conv_fn=getattr(F,f"conv{ndims}d")
        I2,J2,IJ=I*I,J*J,I*J
        I_sum=conv_fn(I,sum_filt,stride=stride,padding=padding)
        J_sum=conv_fn(J,sum_filt,stride=stride,padding=padding)
        I2_sum=conv_fn(I2,sum_filt,stride=stride,padding=padding)
        J2_sum=conv_fn(J2,sum_filt,stride=stride,padding=padding)
        IJ_sum=conv_fn(IJ,sum_filt,stride=stride,padding=padding)
        win_size=float(np.prod(win))
        u_I,u_J=I_sum/win_size,J_sum/win_size
        cross=IJ_sum-u_J*I_sum-u_I*J_sum+u_I*u_J*win_size
        I_var=I2_sum-2*u_I*I_sum+u_I*u_I*win_size
        J_var=J2_sum-2*u_J*J_sum+u_J*u_J*win_size
        return -torch.mean(cross*cross/(I_var*J_var+1e-5))


def parse_args():
    p=argparse.ArgumentParser(description=__doc__)
    add_data_args(p)
    add_geometry_args(p)
    p.add_argument("--stage",choices=["registered","mixed"],default="registered")
    p.add_argument("--epochs",type=int,default=200)
    p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--step_size",type=int,default=20)
    p.add_argument("--lr_gamma",type=float,default=0.8)
    p.add_argument("--validation_interval",type=int,default=1)
    p.add_argument("--seed",type=int,default=1001)
    p.add_argument("--checkpoint_dir",default="")
    p.add_argument("--log_dir",default="")
    p.add_argument("--resume",default="")
    p.add_argument("--init_checkpoint",default="")
    p.add_argument("--max_train_batches",type=int,default=0,
                   help="Optional smoke/debug cap; 0 means all batches.")
    return p.parse_args()


def run_validation(model, loader, p0, args, device, generator, ncc):
    model.eval()
    metrics=MetricAverager()
    l1_values=[]
    with torch.no_grad():
        for batch in loader:
            gt=batch["gt"].to(device,non_blocking=True)
            msi=batch["hr_msi"].to(device,non_blocking=True)
            lr,_=deformed_observation(gt,p0,args,generator,mixed=args.stage=="mixed")
            pred=model(msi,lr,gt.shape[-1])[0]
            l1_values.append(float(F.l1_loss(pred,gt).item()))
            metrics.update(calc_metrics(pred,gt,args.scale_ratio))
    result=metrics.average()
    result["L1"]=float(np.mean(l1_values)) if l1_values else float("nan")
    return result


def main():
    args=parse_args()
    if args.epochs<1 or args.batch_size<1 or args.validation_interval<1:
        raise ValueError("epochs, batch_size and validation_interval must be positive")
    set_seed(args.seed)
    device=resolve_device(args.device)
    cfg=build_shared_cfg(args)
    train_set,val_set,test_set,info=build_datasets(cfg,include_validation=True)
    train_loader=DataLoader(train_set,batch_size=args.batch_size,shuffle=True,
                            num_workers=args.num_workers,pin_memory=device.type=="cuda",
                            drop_last=True)
    val_loader=DataLoader(val_set,batch_size=1,shuffle=False,num_workers=0)
    p0=train_set.degradation_operator.to(device)

    model_args=SimpleNamespace(
        bands=int(info["n_bands"]),
        msi_channels=int(info["n_select_bands"]),
        gpu=(device.index or 0) if device.type=="cuda" else 0,
    )
    model=Main(bands=model_args.bands,args=model_args).to(device)
    optimizer=torch.optim.Adam(model.parameters(),lr=args.lr)
    scheduler=torch.optim.lr_scheduler.StepLR(
        optimizer,step_size=args.step_size,gamma=args.lr_gamma
    )
    ncc=NCC()

    family="physical" if args.stage=="registered" else "hsi_warp_final"
    ckpt_dir=Path(args.checkpoint_dir or HERE/"checkpoints"/family/args.dataset)
    log_dir=Path(args.log_dir or HERE/"logs"/family/args.dataset)
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    log_dir.mkdir(parents=True,exist_ok=True)
    history=log_dir/"validation_history.csv"
    start_epoch=0
    best_psnr=-float("inf")
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        if state.get("method")!="IR-ArF" or state.get("stage")!=args.stage:
            raise ValueError("Checkpoint method/stage mismatch")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch=int(state["epoch"])+1
        best_psnr=float(state.get("best_psnr",-float("inf")))
        print(f"Resumed {args.resume} at epoch {start_epoch+1}")
    elif args.stage=="mixed":
        init=args.init_checkpoint or str(HERE/"checkpoints"/"physical"/args.dataset/"best.pth.tar")
        state=torch.load(init,map_location=device,weights_only=False)
        if state.get("method")!="IR-ArF" or state.get("stage")!="registered":
            raise ValueError("Mixed training requires an IR&ArF registered checkpoint")
        model.load_state_dict(state["model"])
        print(f"Initialized mixed stage from {init}")

    generator=make_generator(device,args.seed+404)
    val_generator=make_generator(device,args.seed+909)
    protocol={
        "method":"IR&ArF",
        "upstream":"https://github.com/Jiahuiqu/IR-ArF",
        "upstream_paper_doi":"10.1109/TIP.2025.3551531",
        "stage":args.stage,
        "dataset":args.dataset,
        "hsi_bands":model_args.bands,
        "msi_bands":model_args.msi_channels,
        "scale_ratio":args.scale_ratio,
        "observation":"LR-HSI=P0(W_phi(X)); HR-MSI=R0(X); GT=X; identity uses P0(X)",
        "deformation_on":"HR-HSI before spatial degradation",
        "optimizer":"Adam",
        "learning_rate":args.lr,
        "scheduler":f"StepLR(step_size={args.step_size}, gamma={args.lr_gamma})",
        "loss":"L1 + 0.1 * mean(six official NCC registration-consistency terms)",
        "arguments":vars(args),
        "data_info":{k:v for k,v in info.items() if k not in ("srf_weights","hsi_wavelengths")},
    }
    (ckpt_dir/"run_protocol.txt").write_text(json.dumps(protocol,indent=2,default=str),encoding="utf-8")
    print(json.dumps(protocol,indent=2,default=str))

    if not history.exists() or start_epoch==0:
        with history.open("w",newline="",encoding="utf-8") as f:
            csv.writer(f).writerow(["epoch","train_loss","train_l1","train_ncc","identity_fraction",
                                    "val_PSNR","val_SAM","val_SSIM","val_ERGAS","val_CC","val_L1"])

    for epoch in range(start_epoch,args.epochs):
        model.train()
        loss_sum=l1_sum=ncc_sum=0.0
        seen=identity_count=0
        bar=tqdm(train_loader,desc=f"IR&ArF {args.stage} {epoch+1}/{args.epochs}",dynamic_ncols=True)
        for step,batch in enumerate(bar):
            if args.max_train_batches and step>=args.max_train_batches:
                break
            gt=batch["gt"].to(device,non_blocking=True)
            msi=batch["hr_msi"].to(device,non_blocking=True)
            lr,identities=deformed_observation(
                gt,p0,args,generator,mixed=args.stage=="mixed"
            )
            outputs=model(msi,lr,gt.shape[-1])
            pred,bx1,bx2,bx3,eh1,eh2,eh3=outputs
            l1=F.l1_loss(pred,gt)
            lr_gray=lr.mean(dim=1,keepdim=True)
            gt_gray=gt.mean(dim=1,keepdim=True)
            ncc_terms=[
                ncc.loss(lr_gray,bx1.mean(dim=1,keepdim=True)),
                ncc.loss(lr_gray,bx2.mean(dim=1,keepdim=True)),
                ncc.loss(lr_gray,bx3.mean(dim=1,keepdim=True)),
                ncc.loss(gt_gray,eh1.mean(dim=1,keepdim=True)),
                ncc.loss(gt_gray,eh2.mean(dim=1,keepdim=True)),
                ncc.loss(gt_gray,eh3.mean(dim=1,keepdim=True)),
            ]
            ncc_loss=torch.stack(ncc_terms).mean()
            loss=l1+0.1*ncc_loss
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Non-finite IR&ArF loss at epoch={epoch+1}, step={step}")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            bsz=gt.shape[0]
            seen+=bsz
            identity_count+=int(identities)
            loss_sum+=float(loss.item())*bsz
            l1_sum+=float(l1.item())*bsz
            ncc_sum+=float(ncc_loss.item())*bsz
            bar.set_postfix(loss=f"{loss_sum/seen:.5f}",l1=f"{l1_sum/seen:.5f}",
                            identity=f"{identity_count/seen:.2f}",refresh=False)
        scheduler.step()
        train_loss=loss_sum/max(seen,1)
        train_l1=l1_sum/max(seen,1)
        train_ncc=ncc_sum/max(seen,1)
        print(f"TRAIN epoch={epoch+1} loss={train_loss:.6f} l1={train_l1:.6f} "
              f"ncc={train_ncc:.6f} identity_fraction={identity_count/max(seen,1):.3f}")

        val_metrics={}
        if (epoch+1)%args.validation_interval==0:
            val_metrics=run_validation(model,val_loader,p0,args,device,val_generator,ncc)
            print("VALIDATION",{"epoch":epoch+1,**val_metrics})
            if val_metrics["PSNR"]>best_psnr:
                best_psnr=val_metrics["PSNR"]
                torch.save({
                    "method":"IR-ArF","stage":args.stage,"epoch":epoch,
                    "model":model.state_dict(),"optimizer":optimizer.state_dict(),
                    "scheduler":scheduler.state_dict(),"best_psnr":best_psnr,
                    "args":vars(args),"model_args":vars(model_args),
                },ckpt_dir/"best.pth.tar")
            with history.open("a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch+1,train_loss,train_l1,train_ncc,identity_count/max(seen,1),
                    val_metrics.get("PSNR"),val_metrics.get("SAM"),val_metrics.get("SSIM"),
                    val_metrics.get("ERGAS"),val_metrics.get("CC"),val_metrics.get("L1")
                ])
        torch.save({
            "method":"IR-ArF","stage":args.stage,"epoch":epoch,
            "model":model.state_dict(),"optimizer":optimizer.state_dict(),
            "scheduler":scheduler.state_dict(),"best_psnr":best_psnr,
            "args":vars(args),"model_args":vars(model_args),
        },ckpt_dir/"last.pth.tar")
    print(f"Training finished. best_PSNR={best_psnr:.4f}; checkpoints={ckpt_dir}")


if __name__=="__main__":
    main()
