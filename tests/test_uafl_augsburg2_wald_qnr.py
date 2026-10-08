"""Unit tests for the explicitly defined MSI-projected QNR adaptation."""
import sys
from pathlib import Path

import numpy as np
import pytest


ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"comparison"/"UAFL"))
from augsburg2_wald_qnr import projected_qnr, _masked_uiqi


def test_uiqi_identity_is_one_even_for_constant_reflectance():
    x=np.full((48,48),.23,dtype=np.float32)
    assert _masked_uiqi(x,x,np.ones_like(x,dtype=bool),48)==pytest.approx(1.,abs=1e-9)


def test_projected_qnr_equals_one_for_perfect_blockwise_sensor_agreement():
    # Each 30m pixel repeats its exact four-band S2 projection on a 3x3
    # grid. This is the ideal, strictly matched synthetic sanity case.
    rng=np.random.default_rng(19)
    h=w=16
    lr=(.12+.7*rng.random((h,w,242))).astype(np.float32)
    r=rng.random((4,242)).astype(np.float64)
    r/=r.sum(axis=1,keepdims=True)
    hr=np.repeat(np.repeat(lr,3,axis=0),3,axis=1)
    s2=(hr@r.T).astype(np.float32)
    mask=np.ones((48,48),dtype=np.uint8)
    result=projected_qnr(hr,lr,s2,mask,r,window_hr=48)
    assert result["QNR"]==pytest.approx(1.,abs=3e-5)
    assert result["Dlambda"]<3e-5
    assert result["Ds"]<3e-5
    assert result["spectral_pair_count"]==6
    assert result["spatial_pair_count"]==16


def test_projected_qnr_rejects_shape_mismatch():
    h=np.zeros((48,48,242),dtype=np.float32)
    lr=np.zeros((16,16,242),dtype=np.float32)
    s2=np.zeros((48,48,4),dtype=np.float32)
    mask=np.ones((48,48),dtype=bool)
    with pytest.raises(ValueError,match="Wald arrays"):
        projected_qnr(h,lr,s2,mask,np.ones((3,242)))
