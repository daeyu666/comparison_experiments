"""Two-optimizer-step smoke test for IR&ArF x4 with configurable channels."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import torch
from model import Main


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hsi_channels",type=int,default=103)
    parser.add_argument("--msi_channels",type=int,default=4)
    parser.add_argument("--hr_size",type=int,default=64)
    parser.add_argument("--device",default="cuda:0" if torch.cuda.is_available() else "cpu")
    args=parser.parse_args()
    if args.hsi_channels<1 or args.msi_channels<1 or args.hr_size<32 or args.hr_size%16:
        raise ValueError("Channel counts must be positive and HR spatial size divisible by 16 (>=32)")
    device=torch.device(args.device)
    model_args=SimpleNamespace(bands=args.hsi_channels,msi_channels=args.msi_channels,
                               gpu=device.index or 0)
    model=Main(bands=args.hsi_channels,args=model_args).to(device)
    lr=torch.rand(1,args.hsi_channels,args.hr_size//4,args.hr_size//4,device=device)
    msi=torch.rand(1,args.msi_channels,args.hr_size,args.hr_size,device=device)
    # A single backward pass can hide an autograd graph retained by a
    # persistent non-leaf attribute (as in the upstream FusionM.eta).
    eta_names=[name for name,_ in model.named_parameters() if name.endswith(".eta")]
    if len(eta_names)!=3:
        raise AssertionError(f"Expected three registered trainable fusion step sizes, found {eta_names}")
    for name, parameter in model.named_parameters():
        if name.endswith(".eta") and (not parameter.is_leaf or parameter.grad_fn is not None):
            raise AssertionError(f"{name} must be a registered leaf Parameter")
    optimizer=torch.optim.Adam(model.parameters(), lr=1e-5)
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        outputs=model(msi,lr,args.hr_size)
        pred=outputs[0]
        expected=(1,args.hsi_channels,args.hr_size,args.hr_size)
        assert pred.shape==expected,(pred.shape,expected)
        assert bool(torch.isfinite(pred).all()),f"nonfinite forward at training step {step+1}"
        loss=pred.abs().mean()
        loss.backward()
        grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
        assert grads and all(bool(torch.isfinite(g).all()) for g in grads), f"nonfinite gradient at step {step+1}"
        optimizer.step()
        print(f"training step {step+1}/2: loss={float(loss.item()):.6f}")
    print("IR&ArF smoke test passed")
    print("device:",device,"input:",tuple(lr.shape),tuple(msi.shape),
          "output:",tuple(pred.shape))
    print("registered fusion step sizes:",eta_names)
    print("parameters:",sum(p.numel() for p in model.parameters()))


if __name__=="__main__":
    main()
