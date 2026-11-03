"""Forward/backward smoke test for IR&ArF x4 with configurable spectral channels."""
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
    outputs=model(msi,lr,args.hr_size)
    pred=outputs[0]
    expected=(1,args.hsi_channels,args.hr_size,args.hr_size)
    assert pred.shape==expected,(pred.shape,expected)
    assert bool(torch.isfinite(pred).all()),"IR&ArF forward produced nonfinite output"
    loss=pred.abs().mean()
    loss.backward()
    grads=[p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(bool(torch.isfinite(g).all()) for g in grads)
    print("IR&ArF smoke test passed")
    print("device:",device,"input:",tuple(lr.shape),tuple(msi.shape),
          "output:",tuple(pred.shape))
    print("parameters:",sum(p.numel() for p in model.parameters()))


if __name__=="__main__":
    main()
