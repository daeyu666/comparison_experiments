"""CPU/CUDA forward, observation loss, backward and six-sensor fit smoke test."""
import argparse
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import torch
from comparison.NonRegSRNet.fit import fit_pair
from comparison.NonRegSRNet.model import FitConfig


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device",default="cpu")
    args=p.parse_args(argv)
    torch.set_num_threads(1)
    device=torch.device(args.device)
    config=FitConfig(iterations=2)
    for name,c,m in [("PaviaU",103,4),("Houston13",144,8),("Chikusei",128,8),
                     ("CAVE",31,3),("Botswana",145,8),("Augsburg",242,4)]:
        generator=torch.Generator(device="cpu").manual_seed(10)
        lr=torch.rand(1,c,4,4,generator=generator).to(device)*0.7+0.1
        msi=torch.rand(1,m,16,16,generator=generator).to(device)*0.7+0.1
        srf=torch.rand(m,c,generator=generator)
        srf=srf/srf.sum(1,keepdim=True)
        prediction,report=fit_pair(lr,msi,srf,4,config)
        assert prediction.shape==(1,c,16,16)
        assert torch.isfinite(prediction).all()
        assert report["gt_used_for_optimization"] is False
        print(f"{name}: shape={list(prediction.shape)} loss={report['last_loss']:.6f} "
              f"parameters={report['parameters']} PASS",flush=True)


if __name__=="__main__":
    main()
