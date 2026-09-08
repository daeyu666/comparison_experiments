"""Strict smoke test for the UAFL comparison reproduction.

Checks:
  1) paper-sized 3-channel model construction and parameter count;
  2) dynamic 4-channel PaviaU reference interface;
  3) registered forward + L1 backward on 103-band 64x64 HSI;
  4) reference-side translation/local deformation forwards;
  5) finite CFDA flow/similarity diagnostics and finite parameter gradients.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from degradations.misalignment import (  # noqa: E402
    MisalignmentParameters,
    apply_misalignment,
    make_misaligned_msi,
)
from dcn_compat import DCN_Refine_With_Prior_flow  # noqa: E402
from model import PAPER_PARAMETER_M, build_uafl, parameter_count  # noqa: E402


def _assert_finite(name, x):
    if not torch.isfinite(x).all():
        raise AssertionError(f"{name} contains non-finite values")


def main():
    if not torch.cuda.is_available():
        raise RuntimeError(
            "UAFL smoke test should be run on CUDA because the production path "
            "uses torchvision's CUDA deform_conv2d backend."
        )
    device = torch.device("cuda:0")
    torch.manual_seed(1234)
    torch.cuda.manual_seed_all(1234)

    # Architecture audit in the original paper's 3-channel reference setting.
    paper_model = build_uafl(3)
    paper_params = parameter_count(paper_model)
    print(
        f"paper-interface params={paper_params:,} ({paper_params/1e6:.4f}M), "
        f"paper report≈{PAPER_PARAMETER_M:.2f}M"
    )
    # The released source and the publication differ slightly in exact accounting,
    # but the paper-sized configuration must remain in the same ~5.94M regime.
    if abs(paper_params / 1e6 - PAPER_PARAMETER_M) > 0.20:
        raise AssertionError(
            "UAFL architecture parameter count is too far from the paper report; "
            "do not train until the architecture audit is resolved"
        )
    del paper_model

    model = build_uafl(4).to(device)
    model.train()
    b, c_hsi, h, w = 1, 103, 64, 64
    lr_hsi = torch.rand(b, c_hsi, h // 4, w // 4, device=device)
    x_up = F.interpolate(lr_hsi, size=(h, w), mode="bicubic", align_corners=False)
    hr_msi = torch.rand(b, 4, h, w, device=device)
    gt = torch.rand(b, c_hsi, h, w, device=device)

    pred = model(x_up, hr_msi)
    if pred.shape != gt.shape:
        raise AssertionError(f"prediction shape {pred.shape} != GT {gt.shape}")
    _assert_finite("registered prediction", pred)
    loss = F.l1_loss(pred, gt)
    _assert_finite("loss", loss)
    loss.backward()

    n_grad = 0
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        n_grad += 1
        _assert_finite(f"grad:{name}", p.grad)
    if n_grad == 0:
        raise AssertionError("no parameter gradients were produced")

    cfdas = [m for m in model.modules() if isinstance(m, DCN_Refine_With_Prior_flow)]
    if not cfdas:
        raise AssertionError("no CFDA module found")
    for i, m in enumerate(cfdas):
        if m.last_prior_flow is None or m.last_prior_similarity is None:
            raise AssertionError(f"CFDA[{i}] did not produce flow/similarity")
        _assert_finite(f"CFDA[{i}] flow", m.last_prior_flow)
        _assert_finite(f"CFDA[{i}] sim", m.last_prior_similarity)

    model.eval()
    with torch.no_grad():
        # Exact 2 px translation on the HR-MSI only.
        zeros = torch.zeros(b, device=device, dtype=hr_msi.dtype)
        local0 = torch.zeros(b, 2, h, w, device=device, dtype=hr_msi.dtype)
        params = MisalignmentParameters(
            dx_px=torch.full((b,), 2.0, device=device),
            dy_px=zeros,
            rotation_deg=zeros,
            local_displacement_px=local0,
        )
        msi_t, valid_t = apply_misalignment(hr_msi, params)
        pred_t = model(x_up, msi_t)
        _assert_finite("translation prediction", pred_t)
        _assert_finite("translation valid", valid_t)

        # Smooth local deformation on HR-MSI only.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(4321)
        msi_l, valid_l, _ = make_misaligned_msi(
            hr_msi,
            local_max_displacement_px=3.0,
            control_grid_size=5,
            generator=gen,
        )
        pred_l = model(x_up, msi_l)
        _assert_finite("local prediction", pred_l)
        _assert_finite("local valid", valid_l)

    print("UAFL smoke test: PASS")
    print(f"registered pred={tuple(pred.shape)} loss={float(loss.item()):.6f}")
    print(
        f"translation valid_fraction={(valid_t >= 0.999).float().mean().item():.4f} "
        f"local valid_fraction={(valid_l >= 0.999).float().mean().item():.4f}"
    )
    print(f"CFDA modules={len(cfdas)} finite_grad_tensors={n_grad}")


if __name__ == "__main__":
    main()
