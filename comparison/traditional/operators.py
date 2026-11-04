"""Exact shared zero-padded optical blur + detector integration and its adjoint."""
import math
import numpy as np
from scipy.ndimage import convolve1d


class PhysicalOperator:
    def __init__(self, scale=4, mtf=.2, truncate=3.):
        if scale < 1 or not 0 < mtf < 1 or truncate <= 0:
            raise ValueError("Invalid physical operator")
        self.scale=int(scale)
        self.sigma=self.scale/math.pi*math.sqrt(-2*math.log(mtf))
        radius=math.ceil(truncate*self.sigma)
        axis=np.arange(-radius,radius+1,dtype=np.float64)
        self.kernel=np.exp(-axis**2/(2*self.sigma**2))
        self.kernel/=self.kernel.sum()

    def blur(self,x):
        return convolve1d(convolve1d(x,self.kernel,axis=-1,mode="constant",cval=0),
                          self.kernel,axis=-2,mode="constant",cval=0)

    def forward(self,x):
        x=np.asarray(x,dtype=np.float64);s=self.scale
        if x.shape[-2]%s or x.shape[-1]%s:
            raise ValueError("Spatial dimensions must be divisible by scale")
        z=self.blur(x)
        return z.reshape(*z.shape[:-2],z.shape[-2]//s,s,z.shape[-1]//s,s).mean(axis=(-1,-3))

    def adjoint(self,y):
        s=self.scale
        z=np.repeat(np.repeat(np.asarray(y,dtype=np.float64),s,axis=-2),s,axis=-1)/(s*s)
        return self.blur(z)


def validate_pair(lr,msi,srf,operator):
    lr=np.array(lr,dtype=np.float64,copy=True);msi=np.array(msi,dtype=np.float64,copy=True)
    r=np.asarray(srf,dtype=np.float64)
    if lr.ndim!=3 or msi.ndim!=3:
        raise ValueError("Expected CHW observations")
    if tuple(v*operator.scale for v in lr.shape[-2:])!=msi.shape[-2:]:
        raise ValueError("Observation scale mismatch")
    if r.shape!=(msi.shape[0],lr.shape[0]) or (r<0).any() or (r.sum(1)<=0).any():
        raise ValueError("Invalid shared SRF")
    if not all(np.isfinite(v).all() for v in (lr,msi,r)):
        raise ValueError("Observations and SRF must be finite")
    if min(lr.min(),msi.min()) < -1e-7:
        raise ValueError("Traditional benchmark expects nonnegative observations")
    return np.maximum(lr,0),np.maximum(msi,0),r

