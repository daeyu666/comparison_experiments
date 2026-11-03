"""Small CPU/CUDA tensor sanity test for the SA-RFM adapter (no dataset files)."""
from __future__ import annotations

import argparse
import torch

from model import SARFMModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hsi_channels", type=int, default=103)
    parser.add_argument("--msi_channels", type=int, default=4)
    parser.add_argument("--dim", type=int, default=220)
    parser.add_argument("--full", action="store_true",
                        help="Also check 128x128 evaluation without tiling")
    args = parser.parse_args()
    device = torch.device(args.device)
    torch.manual_seed(123)

    model = SARFMModel(args.hsi_channels, args.msi_channels,
                       dim=args.dim, ode_steps=1).to(device)
    lr = torch.rand(1, args.hsi_channels, 16, 16, device=device)
    gt = torch.rand(1, args.hsi_channels, 64, 64, device=device)
    msi = torch.rand(1, args.msi_channels, 64, 64, device=device)

    model.train()
    loss, parts = model.flow_loss(lr, gt, msi, lambda_otd=1.0)
    assert torch.isfinite(loss), "Loss is not finite"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "No finite backward gradients"
    print(f"SA-RFM 64x64 train OK: loss={loss.item():.6f} "
          f"RFM={parts['rfm'].item():.6f} OTD={parts['otd'].item():.6f}")

    model.eval()
    with torch.inference_mode():
        pred = model(lr, msi)
        assert pred.shape == gt.shape and torch.isfinite(pred).all()
        print(f"SA-RFM Euler 64x64 OK: {tuple(pred.shape)}")
        if args.full:
            lr2 = torch.rand(1, args.hsi_channels, 32, 32, device=device)
            msi2 = torch.rand(1, args.msi_channels, 128, 128, device=device)
            out2 = model(lr2, msi2)
            assert out2.shape == (1, args.hsi_channels, 128, 128)
            assert torch.isfinite(out2).all()
            print(f"SA-RFM 128x128 full-image inference OK: {tuple(out2.shape)}")


if __name__ == "__main__":
    main()
