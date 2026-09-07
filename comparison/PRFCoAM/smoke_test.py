"""CUDA forward/backward smoke test for the adapted PRFCoAM model."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model_adapter import (  # noqa: E402
    build_prfcoam,
    displacement_smoothness,
    unpack_outputs,
)


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("PRFCoAM smoke test requires CUDA")
    device = torch.device("cuda:0")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    model = build_prfcoam(103, 4, device)
    model.train()
    lr_hsi = torch.rand(1, 103, 16, 16, device=device)
    hr_msi = torch.rand(1, 4, 64, 64, device=device)
    gt = torch.rand(1, 103, 64, 64, device=device)

    outputs = model(lr_hsi, hr_msi)
    pred, pred_msi, pred_lrms, reg_lrhs, rg1, rg2 = unpack_outputs(outputs)
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert pred_msi.shape == hr_msi.shape, (pred_msi.shape, hr_msi.shape)
    assert pred_lrms.shape[-2:] == lr_hsi.shape[-2:]
    assert reg_lrhs.shape == lr_hsi.shape
    assert rg1.shape[1] == 2 and rg2.shape[1] == 2

    recon = F.l1_loss(pred, gt)
    sensor = F.l1_loss(pred_msi, hr_msi)
    smooth = 0.5 * (displacement_smoothness(rg1) + displacement_smoothness(rg2))
    loss = 1.1 * recon + 0.1 * sensor + 0.01 * smooth
    loss.backward()

    grad_tensors = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    if not grad_tensors:
        raise RuntimeError("no gradients were produced")
    if not all(torch.isfinite(g).all().item() for g in grad_tensors):
        raise RuntimeError("non-finite gradient detected")

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("PRFCoAM smoke test: PASS")
    print(f"pred={tuple(pred.shape)} pred_msi={tuple(pred_msi.shape)}")
    print(f"loss={loss.item():.6f} trainable_params={params:,}")
    print(
        f"RG1_mean={torch.linalg.vector_norm(rg1, dim=1).mean().item():.4f}px "
        f"RG2_mean={torch.linalg.vector_norm(rg2, dim=1).mean().item():.4f}px"
    )


if __name__ == "__main__":
    main()
