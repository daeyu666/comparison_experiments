"""Fail-fast diagnostics and explicit optimizer reconfiguration for PMI-RFCoNet."""
import json
from pathlib import Path
import torch


def tensor_stats(value):
    value=value.detach()
    finite=torch.isfinite(value)
    selected=value[finite].float()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
        nonfinite=int((~finite).sum().item()),
        min=float(selected.min()) if selected.numel() else None,
        max=float(selected.max()) if selected.numel() else None)


def first_nonfinite_module(model,lr_hsi,msi):
    """Replay only after failure; preserve normalization buffers and caller RNG."""
    handles=[];buffers={name:value.detach().clone() for name,value in model.named_buffers()}
    device=lr_hsi.device
    devices=[device.index if device.index is not None else torch.cuda.current_device()] if device.type=="cuda" else []
    class Located(Exception):
        pass
    def hook(name):
        def inspect(_module,_inputs,output):
            values=output if isinstance(output,(list,tuple)) else (output,)
            if any(torch.is_tensor(v) and not bool(torch.isfinite(v).all()) for v in values):
                raise Located(name)
        return inspect
    try:
        for name,module in model.named_modules():
            if name:
                handles.append(module.register_forward_hook(hook(name)))
        with torch.random.fork_rng(devices=devices),torch.no_grad():
            model(lr_hsi,msi)
    except Located as exc:
        return str(exc)
    except Exception as exc:
        return f"Replay raised {type(exc).__name__}: {exc}"
    finally:
        for handle in handles:
            handle.remove()
        with torch.no_grad():
            for name,value in model.named_buffers():
                value.copy_(buffers[name])
    return None


def fail_training(model,optimizer,lr_hsi,msi,gt,*,directory,args,epoch,batch_idx,reason,replay=True):
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    report=dict(reason=reason,epoch=epoch,batch_idx=batch_idx,
        learning_rates=[group["lr"] for group in optimizer.param_groups],
        inputs={name:tensor_stats(value) for name,value in
            (("lr_hsi",lr_hsi),("hr_msi",msi),("gt",gt))},
        first_nonfinite_module=first_nonfinite_module(model,lr_hsi,msi) if replay else None,
        invalid_state=[name for name,value in model.state_dict().items()
                       if not bool(torch.isfinite(value).all())],
        gradients={name:tensor_stats(value.grad) for name,value in model.named_parameters()
                   if value.grad is not None},
        args=vars(args).copy())
    (directory/"failure_report.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    # Debug snapshot is separate from best/last; it is never a resume checkpoint.
    torch.save(dict(report=report,model={k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
        lr_hsi=lr_hsi.detach().cpu(),hr_msi=msi.detach().cpu(),gt=gt.detach().cpu()),
        directory/"failure_snapshot.pth.tar")
    raise FloatingPointError(f"PMI-RFCoNet {reason} at epoch={epoch} batch={batch_idx}; "
        f"first_nonfinite_module={report['first_nonfinite_module']}; diagnostics={directory/'failure_report.json'}")


def ensure_finite_checkpoint(state):
    for section in ("model","optimizer"):
        def inspect(value,path):
            if torch.is_tensor(value) and not bool(torch.isfinite(value).all()):
                raise ValueError(f"Cannot resume nonfinite checkpoint tensor {path}")
            if isinstance(value,dict):
                for key,item in value.items():
                    inspect(item,f"{path}.{key}")
            elif isinstance(value,(list,tuple)):
                for key,item in enumerate(value):
                    inspect(item,f"{path}.{key}")
        inspect(state.get(section,{}),section)


def reconfigure_learning_rate(optimizer,scheduler,new_base_lr):
    """Keep Adam moments and scheduler phase; change only the base learning rate."""
    state=scheduler.state_dict()
    old_base=list(state["base_lrs"])
    current=[]
    for group,base in zip(optimizer.param_groups,old_base):
        if base<=0:
            raise ValueError("Resume scheduler base_lr must be positive")
        group["lr"]=new_base_lr*(group["lr"]/base)
        group["initial_lr"]=new_base_lr
        current.append(group["lr"])
    state["base_lrs"]=[new_base_lr]*len(old_base)
    state["_last_lr"]=current
    scheduler.load_state_dict(state)


def nonfinite_bn_buffers(model):
    entries=[(name+".running_"+kind,getattr(module,"running_"+kind))
             for name,module in model.named_modules()
             if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)
             for kind in ("mean","var") if getattr(module,"running_"+kind) is not None]
    if not entries:
        return []
    good=torch.stack([torch.isfinite(value).all() for _,value in entries])
    if bool(good.all()):
        return []
    return [name for (name,_),ok in zip(entries,good.tolist()) if not ok]


def repair_bn_checkpoint(state,model):
    """Reset BN buffers only after proving parameters/Adam/other buffers finite."""
    allowed=set()
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            allowed.update(name+"."+key for key in ("running_mean","running_var"))
    invalid=[name for name,value in state["model"].items()
             if torch.is_tensor(value) and not bool(torch.isfinite(value).all())]
    recoverable=[name for name in invalid if name in allowed]
    # Inspect an in-memory view with only recoverable BN buffers omitted.
    view=dict(state)
    view["model"]={k:v for k,v in state["model"].items() if k not in recoverable}
    ensure_finite_checkpoint(view)
    if not recoverable:
        return None
    initial=model.state_dict()
    replaced=[]
    for name,module in model.named_modules():
        if isinstance(module,torch.nn.modules.batchnorm._BatchNorm):
            for suffix in ("running_mean","running_var","num_batches_tracked"):
                key=name+"."+suffix
                if key in state["model"]:
                    template=initial[key].detach().cpu()
                    state["model"][key]=torch.ones_like(template) if suffix=="running_var" else torch.zeros_like(template)
                    replaced.append(key)
    ensure_finite_checkpoint(state)
    return dict(damaged_buffers=recoverable,reset_buffers=replaced,
                scope="BatchNorm statistics only; model parameters and Adam state retained")


@torch.no_grad()
def recalibrate_bn(model,batches,device,observation_builder,max_batches):
    """Rebuild population statistics from training observations, with fixed weights."""
    modules=list(model.modules())
    modes=[module.training for module in modules]
    bn=[module for module in modules if isinstance(module,torch.nn.modules.batchnorm._BatchNorm)]
    momenta=[module.momentum for module in bn]
    model.eval()  # Disable dropout; only BN statistics are updated.
    for module in bn:
        module.train()
        module.momentum=None  # Cumulative mean over calibration minibatches.
    completed=0
    try:
        for batch in batches:
            if completed>=max_batches:
                break
            gt=batch["gt"].to(device)
            msi=batch["hr_msi"].to(device)
            lr=observation_builder(gt)
            if not all(bool(torch.isfinite(value).all()) for value in (lr,msi)):
                raise FloatingPointError("BN calibration observations are nonfinite")
            prediction=model(lr,msi)
            invalid=nonfinite_bn_buffers(model)
            if invalid or not bool(torch.isfinite(prediction).all()):
                raise FloatingPointError(
                    f"BN recalibration failed at batch={completed}; invalid_buffers={invalid}. "
                    "The current weights cannot be recovered safely this way; resume an earlier finite checkpoint or restart.")
            completed+=1
    finally:
        for module,momentum in zip(bn,momenta):
            module.momentum=momentum
        for module,mode in zip(modules,modes):
            module.training=mode
    if not completed:
        raise ValueError("BN calibration needs at least one training minibatch")
    return completed
