"""Exercise the released width-64 core on all six shared sensor interfaces."""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if not __package__:
    __package__ = "comparison.PMI-RFCoNet"

import torch
from .model import PMIRFCoNet


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--full", action="store_true", help="Also measure a 128x128 forward per sensor")
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    if device.type == "cpu":
        torch.set_num_threads(min(torch.get_num_threads(), 2))
    for name,hsi,msi in (("PaviaU",103,4),("Houston13",144,8),("Chikusei",128,8),
                         ("CAVE",31,3),("Botswana",145,8),("Augsburg",242,4)):
        torch.manual_seed(10)
        model = PMIRFCoNet(hsi,msi).to(device).train()
        lr = torch.rand(1,hsi,16,16,device=device)
        hr = torch.rand(1,msi,64,64,device=device)
        out = model(lr,hr)
        loss = (out-torch.rand_like(out)).abs().mean()
        loss.backward()
        assert out.shape == (1,hsi,64,64) and torch.isfinite(out).all()
        assert all(torch.isfinite(v.grad).all() for v in model.parameters() if v.grad is not None)
        row = dict(dataset=name, shape=list(out.shape), loss=float(loss.detach()),
                   parameters=sum(v.numel() for v in model.parameters()), status="PASS")
        if args.full:
            model.zero_grad(set_to_none=True)
            model.eval()
            lr = torch.rand(1,hsi,32,32,device=device)
            hr = torch.rand(1,msi,128,128,device=device)
            with torch.no_grad():
                model(lr,hr)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                start = time.perf_counter()
                prediction = model(lr,hr)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
            elapsed = time.perf_counter()-start
            assert prediction.shape == (1,hsi,128,128) and torch.isfinite(prediction).all()
            row["forward_128_ms"] = elapsed*1000
        print(json.dumps(row),flush=True)


if __name__ == "__main__":
    main()

