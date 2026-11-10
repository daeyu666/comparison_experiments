"""Model-specific Wald x3 interfaces, without changing the benchmark observations."""
from importlib import import_module

from torch import nn
from torch.nn import functional as F

from comparison.UAFL.augsburg2_wald_common import masked_l1
from .common import pad_pair

HSI, MSI = 242, 4


def resize(x, size):
    return F.interpolate(x, size=size, mode="bilinear", align_corners=False, antialias=True)


class DyadicWaldAdapter(nn.Module):
    """Retain a model's released dyadic stages on the observed LR lattice.

    Only the network's MSI tensor is interpolated onto the 4x internal grid.
    The prediction returns to the 3x observation grid before loss or scoring.
    """
    def __init__(self, network, *, minimum_lr=8):
        super().__init__()
        self.network = network
        self.minimum_lr = minimum_lr

    def forward_full(self, lr, msi):
        proxy_msi = resize(msi, tuple(4 * n for n in lr.shape[-2:]))
        return self.network.forward_full(lr, proxy_msi)

    def forward(self, lr, msi):
        h, w = msi.shape[-2:]
        lr, msi = pad_pair(lr, msi, multiple=self.minimum_lr, minimum=self.minimum_lr)
        proxy_msi = resize(msi, tuple(4 * n for n in lr.shape[-2:]))
        return resize(self.network(lr, proxy_msi), msi.shape[-2:])[..., :h, :w]


class PMI_WaldAdapter(nn.Module):
    def __init__(self, network):
        super().__init__()
        self.network = network

    def forward(self, lr, msi):
        h, w = msi.shape[-2:]
        lr, msi = pad_pair(lr, msi, multiple=8, minimum=16)
        return self.network(lr, msi)[..., :h, :w]


def build_model(method, device):
    module = import_module(f"comparison.{method}.model")
    if method == "MoG-DCN":
        model = module.MoGDCN(HSI, MSI, scale=3)
    elif method == "ESFS":
        model = module.ESFS(HSI, MSI, scale=3)
    elif method == "PSRT":
        model = module.PSRTFusion(HSI, MSI, patch_size=24)
    elif method == "SA-RFM":
        model = module.SARFMModel(HSI, MSI)
    elif method == "M2DTN":
        model = DyadicWaldAdapter(module.build_m2dtn(HSI, MSI), minimum_lr=16)
    elif method == "MoE-PNP":
        model = DyadicWaldAdapter(module.MoEPNP(HSI, MSI, image_size=32))
    elif method == "PMI-RFCoNet":
        model = PMI_WaldAdapter(module.PMIRFCoNet(HSI, MSI, scale_ratio=3))
    else:
        raise ValueError(f"No new Wald training model: {method}")
    return model.to(device)


def architecture_id(method):
    module = import_module(f"comparison.{method}.model")
    name = getattr(module, "ARCHITECTURE_ID", "moe_pnp_dynamic_io_v1")
    variant = ("dyadic_proxy" if method in ("M2DTN", "MoE-PNP") else "native")
    return f"{name}:wald_x3_{variant}_v1"


def training_loss(method):
    if method == "SA-RFM":
        return lambda model, lr, msi, gt, mask: model.flow_loss(lr, gt, msi, valid_mask=mask)[0]
    if method == "MoE-PNP":
        train = import_module("comparison.MoE-PNP.train")
        blur = train.transforms.GaussianBlur(kernel_size=15, sigma=1.0)
        ncc = train._LocalNCC()
        return lambda model, lr, msi, gt, mask: train.compute_moepnp_loss(
            model, lr, msi, gt, blur, ncc, valid_mask=mask,
        )[0]
    return lambda model, lr, msi, gt, mask: masked_l1(model(lr, msi), gt, mask)


def classical_solver(method, cache):
    import math
    import numpy as np
    from threadpoolctl import threadpool_limits
    from comparison.traditional.operators import PhysicalOperator

    class WaldOperator(PhysicalOperator):
        """S2Diff effective Gaussian, zero boundary, then exact 3x area pooling."""
        def __init__(self, sigma):
            self.scale, self.sigma = 3, sigma
            if sigma <= 1e-8:
                self.kernel = np.ones(1, dtype=np.float64)
            else:
                radius = math.ceil(3 * sigma)
                x = np.arange(-radius, radius + 1, dtype=np.float64)
                self.kernel = np.exp(-x**2 / (2 * sigma**2))
                self.kernel /= self.kernel.sum()

    module = import_module(f"comparison.{method}.model")
    op = WaldOperator(cache.sigma)

    def solve(lr, msi):
        # Both released solver interfaces use nonnegative observations. The
        # train-only gain/bias can produce negative MSI values near zero.
        with threadpool_limits(limits=1):
            arr, report = module.fuse(
                np.maximum(lr, 0).transpose(2, 0, 1), np.maximum(msi, 0).transpose(2, 0, 1),
                cache.srf, op,
            )
        return arr.transpose(1, 2, 0).astype(np.float32), report

    return solve
