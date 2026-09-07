"""Small forward/backward smoke test for the HSIFN reproduction."""

import torch
import torch.nn.functional as F

from model import HSIFN


def main():
    torch.manual_seed(1)
    hsi_bands = 8
    msi_bands = 4
    scale = 4
    hr = 32
    lr = hr // scale

    srf = torch.rand(msi_bands, hsi_bands)
    srf = srf / srf.sum(dim=1, keepdim=True)
    model = HSIFN(hsi_bands, msi_bands, srf, use_mask=True)

    gt = torch.rand(1, hsi_bands, hr, hr)
    lr_hsi = F.interpolate(gt, size=(lr, lr), mode="bicubic", align_corners=False)
    hr_msi = torch.einsum("mc,bchw->bmhw", srf, gt)

    pred, diag = model(lr_hsi, hr_msi)
    assert pred.shape == gt.shape, (pred.shape, gt.shape)
    assert diag["coarse_flow"].shape == (1, 2, hr, hr)
    assert len(diag["masks"]) == 4
    loss = F.smooth_l1_loss(pred, gt)
    loss.backward()

    finite_grads = [
        torch.isfinite(p.grad).all().item()
        for p in model.parameters()
        if p.grad is not None
    ]
    assert finite_grads and all(finite_grads), "non-finite gradient detected"
    print(
        "HSIFN smoke test passed: "
        f"loss={loss.item():.6f}, pred={tuple(pred.shape)}, "
        f"params={sum(p.numel() for p in model.parameters()):,}"
    )


if __name__ == "__main__":
    main()
