"""Train the official IR&ArF architecture on the shared HSI-MSI protocol."""
from __future__ import annotations

import argparse
import csv
import json
import hashlib
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
    make_generator, resolve_device, set_seed, snapshot_rng, restore_rng,
    sample_synthetic_geometry, make_deformed_lr_hsi, geometry_kwargs,
    require_srf_weights, DATASET_VALIDATION_INTERVAL,
)
from data_loader import build_datasets  # noqa: E402
from metrics import MetricAverager, calc_metrics  # noqa: E402
from model import Main, UPSTREAM_COMMIT, IRARF_ARCHITECTURE_ID  # noqa: E402


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
    p.add_argument("--epochs",type=int,default=None)
    p.add_argument("--batch_size",type=int,default=1)
    p.add_argument("--num_workers",type=int,default=0)
    p.add_argument("--lr",type=float,default=2e-4)
    p.add_argument("--step_size",type=int,default=20)
    p.add_argument("--lr_gamma",type=float,default=0.8)
    p.add_argument("--grad_clip_norm",type=float,default=1.0,
                   help="Global gradient norm clipping; 0 disables clipping")
    p.add_argument("--max_abs_output",type=float,default=20.0,
                   help="Abort when normalized reconstructed HSI exceeds this magnitude")
    p.add_argument("--warn_abs_output",type=float,default=2.0,
                   help="Create one detailed recurrent-stage warning trace per epoch when exceeded")
    p.add_argument("--validation_interval",type=int,default=None)
    p.add_argument("--validation_cases",type=int,default=5)
    p.add_argument("--seed",type=int,default=1001)
    p.add_argument("--checkpoint_dir",default="")
    p.add_argument("--log_dir",default="")
    p.add_argument("--resume",default="")
    p.add_argument("--init_checkpoint",default="")
    p.add_argument("--max_train_batches",type=int,default=0,
                   help="Optional smoke/debug cap; 0 means all batches.")
    return p.parse_args()


def run_validation(model, loader, p0, args, device):
    """Select registered PSNR for stage 1 and deterministic warp PSNR for stage 2."""
    model.eval()
    reg, warp = MetricAverager(), MetricAverager()
    generator = make_generator(device, args.seed + 60000)
    l1_values = []
    with torch.no_grad():
        for batch in loader:
            gt=batch["gt"].to(device,non_blocking=True)
            msi=batch["hr_msi"].to(device,non_blocking=True)
            registered_lr=p0.degrade(gt)
            pred=model(msi,registered_lr,gt.shape[-1])[0]
            reg.update(calc_metrics(pred,gt,args.scale_ratio))
            l1_values.append(float(F.l1_loss(pred,gt).item()))
            if args.stage=="mixed":
                for _ in range(args.validation_cases):
                    geometry=sample_synthetic_geometry(
                        *gt.shape[-2:], device=device, dtype=gt.dtype,
                        generator=generator,
                        local_strength_min_fraction=0.0,
                        local_strength_max_fraction=1.0,
                        **geometry_kwargs(args),
                    )
                    warped_lr=make_deformed_lr_hsi(gt,geometry,p0)
                    warp_pred=model(msi,warped_lr,gt.shape[-1])[0]
                    warp.update(calc_metrics(warp_pred,gt,args.scale_ratio))
    reg_scores=reg.average()
    chosen=warp.average() if args.stage=="mixed" else reg_scores
    result=dict(chosen)
    result["reg_PSNR"]=reg_scores["PSNR"]
    result["reg_SAM"]=reg_scores["SAM"]
    if args.stage=="mixed":
        result["warp_PSNR"]=chosen["PSNR"]
        result["warp_SAM"]=chosen["SAM"]
    result["L1"]=float(np.mean(l1_values)) if l1_values else float("nan")
    return result


CHECKPOINT_PROTOCOL_KEYS = (
    "dataset", "image_size", "patch_size", "stride", "scale_ratio",
    "degradation_mode", "degradation_sigma", "degradation_kernel_size",
    "mtf_nyquist", "psf_truncate", "msi_mode", "srf_path",
    "wavelength_root", "wavelength_path", "srf_interp", "srf_band_set",
    "max_translation", "max_rotation_deg", "max_local_px",
    "control_grid", "min_jacobian", "registered_probability",
)
RESUME_KEYS = ("seed", "batch_size", "lr", "step_size", "lr_gamma",
               "grad_clip_norm", "max_abs_output", "warn_abs_output",
               "validation_interval", "validation_cases", "max_train_batches")


def verify_train_checkpoint(state, args, model_args, sensor_fingerprint, *, stage, resume=False):
    if state.get("method")!="IR-ArF" or state.get("stage")!=stage:
        raise ValueError(f"IR&ArF checkpoint method/stage mismatch: expected {stage}")
    if state.get("architecture_id")!=IRARF_ARCHITECTURE_ID:
        raise ValueError("IR&ArF checkpoint architecture version mismatch; start a new run")
    if state.get("upstream_commit")!=UPSTREAM_COMMIT:
        raise ValueError("IR&ArF model-source version mismatch")
    if state.get("sensor_fingerprint")!=sensor_fingerprint:
        raise ValueError("IR&ArF sensor SRF mismatch")
    saved_model=state.get("model_args",{})
    for key in ("bands","msi_channels"):
        if saved_model.get(key)!=getattr(model_args,key):
            raise ValueError(f"IR&ArF checkpoint architecture mismatch: {key}")
    recorded=state.get("args")
    if not isinstance(recorded,dict):
        raise ValueError("IR&ArF checkpoint is missing protocol arguments")
    for key in CHECKPOINT_PROTOCOL_KEYS + (RESUME_KEYS if resume else ()):
        if key in recorded and recorded[key]!=getattr(args,key):
            raise ValueError(f"IR&ArF checkpoint protocol mismatch: {key}")


def make_checkpoint(args, model_args, sensor_fingerprint, model, optimizer,
                    scheduler, generator, epoch, best_psnr):
    return {
        "method":"IR-ArF", "stage":args.stage, "epoch":epoch,
        "architecture_id":IRARF_ARCHITECTURE_ID,
        "upstream_commit":UPSTREAM_COMMIT, "sensor_fingerprint":sensor_fingerprint,
        "model":model.state_dict(), "optimizer":optimizer.state_dict(),
        "scheduler":scheduler.state_dict(), "best_psnr":best_psnr,
        "args":vars(args).copy(), "model_args":vars(model_args).copy(),
        "rng":snapshot_rng(generator),
    }



def trace_reconstruction(model, msi, lr, spatial_size, log_dir, *, epoch, batch, kind, threshold):
    """Capture recurrent updates and shared branch amplitudes on a problematic batch.

    It performs a no-grad replay only on warning/failure and does not alter
    model weights, gradients, or the training loss.
    """
    def tensor_stats(value):
        if isinstance(value, torch.Tensor):
            x=value.detach()
            return {
                "shape":list(x.shape),
                "finite":bool(torch.isfinite(x).all().item()),
                "abs_max":float(x.abs().amax().item()),
                "mean":float(x.float().mean().item()),
            }
        if isinstance(value,(tuple,list)):
            return [tensor_stats(v) for v in value]
        return str(type(value).__name__)

    events=[]
    def collect(name):
        def hook(_module, inputs, outputs):
            important_inputs=inputs if name.startswith("F") else inputs[:1]
            events.append({
                "name":name,
                "inputs":tensor_stats(important_inputs),
                "outputs":tensor_stats(outputs),
            })
        return hook

    net=model.FandD
    modules=(
        ("B_T",net.F1.B_T), ("R",net.F1.R), ("R_T",net.F1.R_T),
        ("F1",net.F1), ("F2",net.F2), ("F3",net.F3),
        ("Denoiser_shared",net.D1),
    )
    handles=[module.register_forward_hook(collect(name)) for name,module in modules]
    replay_error=None
    try:
        with torch.no_grad():
            model(msi,lr,spatial_size)
    except Exception as exc:
        replay_error=f"{type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()
    diagnostic={
        "epoch":epoch,"batch":batch,"kind":kind,"threshold":threshold,
        "input_lr_hsi":tensor_stats(lr),"input_hr_msi":tensor_stats(msi),
        "eta":[float(net.F1.eta.item()),float(net.F2.eta.item()),float(net.F3.eta.item())],
        "events":events,"replay_error":replay_error,
    }
    out=log_dir/f"instability_{kind}_epoch{epoch:04d}_batch{batch:05d}.json"
    out.write_text(json.dumps(diagnostic,indent=2,ensure_ascii=False),encoding="utf-8")
    return out


def main():
    args=parse_args()
    if args.epochs is None:
        args.epochs = 800 if args.stage=="registered" else 600
    if args.validation_interval is None:
        args.validation_interval = DATASET_VALIDATION_INTERVAL[args.dataset]
    if args.warn_abs_output<0 or args.warn_abs_output>=args.max_abs_output:
        raise ValueError("warn_abs_output must satisfy 0 <= warn < max_abs_output")
    if args.grad_clip_norm<0 or args.max_abs_output<=0:
        raise ValueError("grad_clip_norm must be >=0 and max_abs_output must be positive")
    if args.validation_cases < 1:
        raise ValueError("validation_cases must be >=1")
    if (args.image_size, args.patch_size, args.stride, args.scale_ratio)!=(128,64,32,4):
        raise ValueError("IR&ArF shared protocol requires 128/64/32/x4")
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
    sensor_fingerprint=hashlib.sha256(
        torch.as_tensor(require_srf_weights(info),dtype=torch.float32).cpu().numpy().tobytes()
    ).hexdigest()

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
    generator=make_generator(device,args.seed+404)

    family="physical" if args.stage=="registered" else "hsi_warp_final"
    if args.max_train_batches:
        family += "_smoke"
        print("DEBUG RUN: limited batches; checkpoint/logs isolated from formal training")
    ckpt_dir=Path(args.checkpoint_dir or HERE/"checkpoints"/family/args.dataset)
    log_dir=Path(args.log_dir or HERE/"logs"/family/args.dataset)
    ckpt_dir.mkdir(parents=True,exist_ok=True)
    log_dir.mkdir(parents=True,exist_ok=True)
    history=log_dir/"validation_history.csv"
    start_epoch=0
    best_psnr=-float("inf")
    if args.resume:
        state=torch.load(args.resume,map_location=device,weights_only=False)
        verify_train_checkpoint(state,args,model_args,sensor_fingerprint,
                                stage=args.stage,resume=True)
        model.load_state_dict(state["model"],strict=True)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch=int(state["epoch"])+1
        best_psnr=float(state.get("best_psnr",-float("inf")))
        if "rng" not in state:
            raise ValueError("IR&ArF resume checkpoint is missing RNG state")
        restore_rng(state["rng"],generator)
        print(f"Resumed {args.resume} at epoch {start_epoch+1}")
    elif args.stage=="mixed":
        init=args.init_checkpoint or str(HERE/"checkpoints"/"physical"/args.dataset/"best.pth.tar")
        state=torch.load(init,map_location=device,weights_only=False)
        verify_train_checkpoint(state,args,model_args,sensor_fingerprint,
                                stage="registered")
        model.load_state_dict(state["model"],strict=True)
        print(f"Initialized mixed stage from {init}")

    protocol={
        "method":"IR&ArF",
        "architecture_id":IRARF_ARCHITECTURE_ID,
        "grad_clip_norm":args.grad_clip_norm,
        "max_abs_output":args.max_abs_output,
        "warn_abs_output":args.warn_abs_output,
        "upstream":"https://github.com/Jiahuiqu/IR-ArF",
        "upstream_commit":UPSTREAM_COMMIT,
        "sensor_fingerprint":sensor_fingerprint,
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
                                    "val_PSNR","val_SAM","val_SSIM","val_ERGAS","val_CC","val_L1",
                                    "grad_norm_max","prediction_abs_max"])

    for epoch in range(start_epoch,args.epochs):
        model.train()
        loss_sum=l1_sum=ncc_sum=0.0
        seen=identity_count=0
        grad_norm_max=0.0
        prediction_abs_max=0.0
        warned_this_epoch=False
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
            with torch.no_grad():
                output_amplitude=float(pred.detach().abs().amax().item())
            prediction_abs_max=max(prediction_abs_max,output_amplitude)
            if not math.isfinite(output_amplitude) or output_amplitude>args.max_abs_output:
                stage_levels=[float(t.detach().abs().amax().item()) for t in outputs]
                diagnostic=trace_reconstruction(
                    model,msi,lr,gt.shape[-1],log_dir,
                    epoch=epoch+1,batch=step+1,kind="failure",
                    threshold=args.max_abs_output,
                )
                raise FloatingPointError(
                    f"IR&ArF reconstruction explosion at epoch={epoch+1}, batch={step+1}, "
                    f"pred_abs_max={output_amplitude:.6g}, stage_abs_max={stage_levels}. "
                    f"Per-stage diagnostic: {diagnostic}. Stop training; do not resume "
                    f"the exploding run."
                )
            if (args.warn_abs_output>0 and output_amplitude>args.warn_abs_output
                    and not warned_this_epoch):
                diagnostic=trace_reconstruction(
                    model,msi,lr,gt.shape[-1],log_dir,
                    epoch=epoch+1,batch=step+1,kind="warning",
                    threshold=args.warn_abs_output,
                )
                print(f"IR&ArF high-amplitude warning pred_abs_max="
                      f"{output_amplitude:.5g}; per-stage trace: {diagnostic}")
                warned_this_epoch=True
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
            # Shared multi-stage fusion/denoising modules can amplify
            # gradients: clip the global norm before updating parameters.
            grad_norm=torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=(args.grad_clip_norm if args.grad_clip_norm>0 else float("inf")),
                error_if_nonfinite=True,
            )
            grad_norm_max=max(grad_norm_max,float(grad_norm.item()))
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
              f"ncc={train_ncc:.6f} identity_fraction={identity_count/max(seen,1):.3f} "
              f"grad_norm_max={grad_norm_max:.4g} pred_abs_max={prediction_abs_max:.4g}")

        val_metrics={}
        if (epoch+1)%args.validation_interval==0 or epoch+1==args.epochs:
            val_metrics=run_validation(model,val_loader,p0,args,device)
            print("VALIDATION",{"epoch":epoch+1,**val_metrics})
            if val_metrics["PSNR"]>best_psnr:
                best_psnr=val_metrics["PSNR"]
                torch.save(
                    make_checkpoint(args,model_args,sensor_fingerprint,model,optimizer,
                                    scheduler,generator,epoch,best_psnr),
                    ckpt_dir/"best.pth.tar",
                )
            with history.open("a",newline="",encoding="utf-8") as f:
                csv.writer(f).writerow([
                    epoch+1,train_loss,train_l1,train_ncc,identity_count/max(seen,1),
                    val_metrics.get("PSNR"),val_metrics.get("SAM"),val_metrics.get("SSIM"),
                    val_metrics.get("ERGAS"),val_metrics.get("CC"),val_metrics.get("L1"),
                    grad_norm_max,prediction_abs_max
                ])
        torch.save(
            make_checkpoint(args,model_args,sensor_fingerprint,model,optimizer,
                            scheduler,generator,epoch,best_psnr),
            ckpt_dir/"last.pth.tar",
        )
    print(f"Training finished. best_PSNR={best_psnr:.4f}; checkpoints={ckpt_dir}")


if __name__=="__main__":
    main()
