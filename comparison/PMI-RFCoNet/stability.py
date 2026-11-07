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

