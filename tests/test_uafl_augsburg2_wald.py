"""Unit guards for UAFL's strict Augsburg-2 Wald data protocol.

Run from comparison_experiments root:
    pytest -q tests/test_uafl_augsburg2_wald.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
UAFL = ROOT / "comparison" / "UAFL"
sys.path.insert(0, str(UAFL))
from augsburg2_wald_common import WaldDataset, predict_uafl, require_wald


class StrictGridModel(torch.nn.Module):
    def forward(self, x, ref):
        assert x.shape[-2] % 16 == 0
        assert x.shape[-1] % 16 == 0
        assert x.shape[-2:] == ref.shape[-2:]
        return x


def _make_cache(root):
    root.mkdir()
    (root / "full").mkdir()
    (root / "full" / "meta.json").write_text(json.dumps({"region":"sub_area_2"}))
    (root / "wald_psf.json").write_text(json.dumps({
        "scale_ratio":3, "terminal_sigma_hr_pixels":1.2,
    }))
    np.save(root/"srf_weights.npy",np.full((4,242),1/242,dtype=np.float32))
    for split in ("train","validation","test"):
        dest=root/split
        dest.mkdir()
        h,w=144,144
        np.save(dest/"gt.npy",np.ones((h,w,242),dtype=np.float32)*.2)
        np.save(dest/"lr_hsi.npy",np.ones((h//3,w//3,242),dtype=np.float32)*.2)
        np.save(dest/"hr_msi.npy",np.ones((h,w,4),dtype=np.float32)*.2)
        np.save(dest/"valid_mask.npy",np.ones((h,w),dtype=np.uint8))
        (dest/"meta.json").write_text(json.dumps({
            "msi_source":"real_Sentinel_2_Wald_30m",
            "target":"30m_EnMAP_like",
            "gt_source":"observed_30m_HSI_only",
            "scale_ratio":3,
        }))


def test_wald_requires_real_msis_and_observed_30m_targets(tmp_path):
    root=tmp_path/"wald"
    _make_cache(root)
    assert require_wald(root)==pytest.approx(1.2)
    meta=root/"validation"/"meta.json"
    obj=json.loads(meta.read_text())
    obj["msi_source"]="synthetic_S2"
    meta.write_text(json.dumps(obj))
    with pytest.raises(ValueError,match="not strict"):
        require_wald(root)


def test_wald_72_training_patch_and_48_eval_patch(tmp_path):
    root=tmp_path/"wald"
    _make_cache(root)
    train=WaldDataset(root,"train",train_patch=72,train_stride=6,eval_patch=48)
    val=WaldDataset(root,"validation",train_patch=72,train_stride=6,eval_patch=48)
    assert len(train)>1
    assert len(val)==9
    assert train[0]["gt"].shape==(242,72,72)
    assert train[0]["lr_hsi"].shape==(242,24,24)
    assert val[0]["gt"].shape==(242,48,48)


def test_uafl_window_padding_crops_back_to_fixed_wald_72():
    model=StrictGridModel()
    lr=torch.rand(1,242,24,24)
    ref=torch.rand(1,4,72,72)
    pred=predict_uafl(model,lr,ref)
    assert pred.shape==(1,242,72,72)
    lr2=torch.rand(1,242,16,16)
    ref2=torch.rand(1,4,48,48)
    assert predict_uafl(model,lr2,ref2).shape==(1,242,48,48)
