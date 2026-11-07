"""Data-free PSRT architecture smoke test (64x64 train; optional 128x128 infer)."""
from __future__ import annotations

import argparse
import torch
import torch.nn.functional as F

from model import PSRTFusion
from model_SR import Win_Shuffle, Win_Reshuffle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--hsi_channels", type=int, default=103)
    parser.add_argument("--msi_channels", type=int, default=4)
    parser.add_argument("--full", action="store_true",
                        help="Also test 128x128 full-image inference")
    args = parser.parse_args()

    torch.manual_seed(123)
    device = torch.device(args.device)
    x = torch.randn(1, 32, 64, 64, device=device)
    for w in (8, 4, 2):
        y = Win_Reshuffle(Win_Shuffle(x, w), w)
        if not torch.equal(x, y):
            raise AssertionError(f"PSRT W2W shuffle/reshuffle is not invertible at window {w}")
    print("PSRT window shuffle/reshuffle identity: PASS", flush=True)

    model = PSRTFusion(args.hsi_channels, args.msi_channels, patch_size=64).to(device)
    print(f"PSRT parameter count={sum(p.numel() for p in model.parameters()):,}", flush=True)
    lr = torch.rand(1, args.hsi_channels, 16, 16, device=device)
    msi = torch.rand(1, args.msi_channels, 64, 64, device=device)
    gt = torch.rand(1, args.hsi_channels, 64, 64, device=device)
    model.train()
    pred = model(lr, msi)
    if pred.shape != gt.shape or not torch.isfinite(pred).all():
        raise AssertionError("PSRT 64x64 forward shape or finite-value failure")
    loss = F.l1_loss(pred, gt)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not gradients or not all(torch.isfinite(g).all() for g in gradients):
        raise AssertionError("PSRT 64x64 backpropagation failed")
    print(f"PSRT 64x64 forward/backward PASS, loss={loss.item():.6f}", flush=True)

    if args.full:
        model.eval()
        with torch.inference_mode():
            full_lr = torch.rand(1, args.hsi_channels, 32, 32, device=device)
            full_msi = torch.rand(1, args.msi_channels, 128, 128, device=device)
            output = model(full_lr, full_msi)
            if output.shape != (1, args.hsi_channels, 128, 128):
                raise AssertionError(f"Unexpected PSRT full-image shape: {output.shape}")
            if not torch.isfinite(output).all():
                raise AssertionError("Nonfinite 128x128 PSRT full-image output")
            print(f"PSRT 128x128 inference PASS: {tuple(output.shape)}", flush=True)


if __name__ == "__main__":
    main()
