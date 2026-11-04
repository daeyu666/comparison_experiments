"""HySure subspace/VTV objective and four-split SALSA/ADMM.

Ported from alfaiate/HySure src/data_fusion.m, commit
8652a5838b95c29ca056f7921f8ac58def6bbaf7 (GPL-3.0).
SVD is the original supported orthonormal subspace variant.
Use the shared P0 and its exact adjoint in place of circular blur/subsampling.
The X normal equation uses conjugate gradients because P0 is nonperiodic.
"""
from dataclasses import dataclass,asdict
import inspect
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.sparse.linalg import LinearOperator,cg
from comparison.traditional.operators import validate_pair

UPSTREAM_COMMIT="8652a5838b95c29ca056f7921f8ac58def6bbaf7"


@dataclass(frozen=True)
class HySureConfig:
    iterations:int=200
    rank:int=10
    lambda_phi:float=5e-4
    lambda_m:float=1.
    mu:float=.05
    cg_rtol:float=1e-6
    cg_maxiter:int=80


def gradient(x):
    return x-np.roll(x,-1,axis=-1),x-np.roll(x,-1,axis=-2)


def divergence(x,y):
    return x-np.roll(x,1,axis=-1)+y-np.roll(y,1,axis=-2)


def objective(x,lr,msi,r,e,operator,config):
    dx,dy=gradient(x)
    return (.5*np.sum((e@operator.forward(x).reshape(e.shape[1],-1)-lr.reshape(lr.shape[0],-1))**2)
        +.5*config.lambda_m*np.sum(((r@e)@x.reshape(e.shape[1],-1)-msi.reshape(msi.shape[0],-1))**2)
        +config.lambda_phi*np.sqrt(np.sum(dx*dx+dy*dy,axis=0)).sum())


def solve_coefficients(lr,msi,r,e,operator,config):
    p=e.shape[1];H,W=msi.shape[-2:];shape=(p,H,W)
    mu=config.mu
    ie=cho_factor(e.T@e+mu*np.eye(p))
    re=r@e;ire=cho_factor(config.lambda_m*(re.T@re)+mu*np.eye(p))
    yyh=e.T@lr.reshape(lr.shape[0],-1)
    yym=config.lambda_m*re.T@msi.reshape(msi.shape[0],-1)
    x=np.zeros(shape)
    v1=np.zeros((p,*lr.shape[-2:]));d1=np.zeros_like(v1)
    v2=np.zeros(shape);v3=np.zeros(shape);v4=np.zeros(shape)
    d2=np.zeros(shape);d3=np.zeros(shape);d4=np.zeros(shape)
    def normal(flat):
        z=flat.reshape(shape);dx,dy=gradient(z)
        return (operator.adjoint(operator.forward(z))+z+divergence(dx,dy)).ravel()
    lin=LinearOperator((x.size,x.size),matvec=normal,dtype=np.float64)
    cg_options=dict(maxiter=config.cg_maxiter,atol=0)
    cg_options["rtol" if "rtol" in inspect.signature(cg).parameters else "tol"]=config.cg_rtol
    for _ in range(config.iterations):
        rhs=operator.adjoint(v1+d1)+v2+d2+divergence(v3+d3,v4+d4)
        solution,status=cg(lin,rhs.ravel(),x0=x.ravel(),**cg_options)
        if status!=0:
            raise RuntimeError(f"HySure normal equation did not converge (CG status={status}); increase cg_maxiter")
        x=solution.reshape(shape)
        nu1=operator.forward(x)-d1
        v1=cho_solve(ie,yyh+mu*nu1.reshape(p,-1)).reshape(v1.shape)
        nu2=x-d2
        v2=cho_solve(ire,yym+mu*nu2.reshape(p,-1)).reshape(shape)
        gx,gy=gradient(x);nu3=gx-d3;nu4=gy-d4
        magnitude=np.sqrt(np.sum(nu3*nu3+nu4*nu4,axis=0))
        shrink=np.maximum(0,1-config.lambda_phi/mu/np.maximum(magnitude,1e-30))
        v3=nu3*shrink;v4=nu4*shrink
        d1=v1-nu1;d2=v2-nu2;d3=v3-nu3;d4=v4-nu4
    return x


def fuse(lr_hsi,hr_msi,srf,operator,config=None):
    cfg=config or HySureConfig()
    if min(cfg.iterations,cfg.rank,cfg.cg_maxiter)<1 or min(cfg.mu,cfg.cg_rtol)<=0:
        raise ValueError("Invalid HySure iteration/rank/ADMM/CG settings")
    if min(cfg.lambda_phi,cfg.lambda_m)<0:
        raise ValueError("HySure regularization weights must be nonnegative")
    lr,msi,r=validate_pair(lr_hsi,hr_msi,srf,operator)
    rank=min(cfg.rank,lr.shape[0],lr.shape[-2]*lr.shape[-1])
    u,_,_=np.linalg.svd(lr.reshape(lr.shape[0],-1),full_matrices=False)
    e=u[:,:rank]
    x=solve_coefficients(lr,msi,r,e,operator,cfg)
    out=(e@x.reshape(rank,-1)).reshape(lr.shape[0],*msi.shape[-2:])
    if not np.isfinite(out).all():
        raise FloatingPointError("HySure produced nonfinite output")
    return out,dict(config=asdict(cfg),effective_rank=rank,basis="SVD",
                    objective=float(objective(x,lr,msi,r,e,operator,cfg)),
                    upstream_commit=UPSTREAM_COMMIT,sensors="known shared SRF/P0")

