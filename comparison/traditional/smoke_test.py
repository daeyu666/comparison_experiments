"""Run both classical solvers on the six sensor interfaces, without any dataset."""
import argparse
import importlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
import numpy as np
from threadpoolctl import threadpool_limits
from comparison.traditional.operators import PhysicalOperator


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full",action="store_true",help="128x128 and default full optimization budgets")
    args=parser.parse_args()
    size=128 if args.full else 32
    rng=np.random.default_rng(13);operator=PhysicalOperator()
    with threadpool_limits(1):
        for name,c,m in (("PaviaU",103,4),("Houston13",144,8),("Chikusei",128,8),
                         ("CAVE",31,3),("Botswana",145,8),("Augsburg",242,4)):
            dictionary=rng.uniform(.05,.8,(c,6))
            abundances=rng.random((6,size,size));abundances/=abundances.sum(0)
            gt=(dictionary@abundances.reshape(6,-1)).reshape(c,size,size)
            r=rng.random((m,c));r/=r.sum(1,keepdims=True)
            lr=operator.forward(gt);msi=(r@gt.reshape(c,-1)).reshape(m,size,size)
            for method in ("CNMF","HySure"):
                module=importlib.import_module("comparison."+method+".model")
                if method=="CNMF":
                    cfg=module.CNMFConfig() if args.full else module.CNMFConfig(10,6)
                else:
                    cfg=module.HySureConfig() if args.full else module.HySureConfig(iterations=10,rank=6)
                start=time.perf_counter();result,_=module.fuse(lr,msi,r,operator,cfg)
                assert result.shape==gt.shape and np.isfinite(result).all()
                print(json.dumps(dict(method=method,dataset=name,shape=list(result.shape),
                    seconds=time.perf_counter()-start,status="PASS")),flush=True)


if __name__=="__main__":
    main()

