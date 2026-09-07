"""Small CPU smoke test for the PSRF reproduction interfaces."""

import torch

from diffusion import PSRFDiffusion


def main():
    model = PSRFDiffusion(
        patch_size=16,
        scale_ratio=4,
        hsi_channels=8,
        msi_channels=4,
        n_timestep=10,
    )
    gt = torch.rand(1, 8, 16, 16)
    lr = torch.rand(1, 8, 4, 4)
    msi = torch.rand(1, 4, 16, 16)
    out = model(gt, lr, msi, generator=torch.Generator().manual_seed(1))
    assert out.prediction.shape == gt.shape
    assert out.coarse_theta.shape == (1, 2, 3)
    assert out.fine_offset_px.shape == (1, 16, 16, 2)

    lr_big = torch.rand(1, 8, 8, 8)
    msi_big = torch.rand(1, 4, 32, 32)
    pred, diag = model.sample_tiled(
        lr_big,
        msi_big,
        sample_steps=2,
        tile_stride=16,
        generator=torch.Generator().manual_seed(2),
    )
    assert pred.shape == (1, 8, 32, 32)
    assert diag["coarse_theta"].shape == (4, 2, 3)
    print("PSRF-DiffNet smoke test passed")


if __name__ == "__main__":
    main()
