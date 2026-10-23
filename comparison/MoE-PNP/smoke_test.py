"""Small CPU/GPU forward smoke test for dynamic MoE-PNP IO."""
import torch
from model import MoEPNP

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = MoEPNP(hsi_channels=8, msi_channels=3, scale_ratio=4, image_size=64).to(device).eval()
    lr_hsi = torch.rand(1, 8, 16, 16, device=device)
    hr_msi = torch.rand(1, 3, 64, 64, device=device)
    with torch.no_grad():
        prediction = model(lr_hsi, hr_msi)
    assert prediction.shape == (1, 8, 64, 64), tuple(prediction.shape)
    assert bool(torch.isfinite(prediction).all())
    print(f"MoE-PNP smoke test passed on {device}: {tuple(prediction.shape)}")

if __name__ == "__main__":
    main()
