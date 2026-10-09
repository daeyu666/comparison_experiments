"""CPU tests for center-heldout Augsburg Region-2 geometry and split provenance."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from augsburg2_wald_center_roi import (
    PROTOCOL, crop_heldout, geotiff_transform, read_roi
)
from augsburg2_wald_common import WaldDataset


class AugsburgCenterHoldoutROITests(unittest.TestCase):
    def create_cache(self, root):
        p=Path(root)
        (p/"roi.json").write_text(json.dumps({
            "protocol_id": PROTOCOL,
            "source_region":"sub_area_2",
            "test_bbox_30m":[24,36,72,84],
            "test_bbox_10m":[72,108,216,252],
            "guard_pixels_30m":6,
        }), encoding="utf-8")

    def test_heldout_crops_exact_same_region_at_both_resolutions(self):
        with TemporaryDirectory() as d:
            self.create_cache(d)
            lr=np.arange(100*120,dtype=np.float32).reshape(100,120,1)
            msi=np.arange(300*360,dtype=np.float32).reshape(300,360,1)
            mask=np.ones((300,360),dtype=np.uint8)
            out=crop_heldout(
                d,lr,msi,mask,ckpt_protocol_id=PROTOCOL,
                ckpt_bbox_30m=[24,36,72,84]
            )
            low,hi,valid,oy,ox,suffix,protocol=out
            self.assertEqual(low.shape,(48,48,1))
            self.assertEqual(hi.shape,(144,144,1))
            self.assertEqual(valid.shape,(144,144))
            self.assertEqual((oy,ox,suffix,protocol),(72,108,"heldout",PROTOCOL))
            self.assertEqual(low[0,0,0],lr[24,36,0])
            self.assertEqual(hi[0,0,0],msi[72,108,0])
            self.assertEqual(hi[-1,-1,0],msi[215,251,0])

    def test_pretrained_full_scene_checkpoint_is_rejected(self):
        with TemporaryDirectory() as d:
            self.create_cache(d)
            a=np.zeros((100,120,242),np.float32)
            b=np.zeros((300,360,4),np.float32)
            c=np.ones((300,360),np.uint8)
            with self.assertRaisesRegex(ValueError,"retrain"):
                crop_heldout(d,a,b,c,ckpt_protocol_id="legacy_full_region_wald")
            with self.assertRaisesRegex(ValueError,"test ROI differs"):
                crop_heldout(d,a,b,c,ckpt_protocol_id=PROTOCOL,
                             ckpt_bbox_30m=[0,0,48,48])

    def test_legacy_does_not_fabricate_a_spatial_split(self):
        with TemporaryDirectory() as d:
            a=np.zeros((100,120,242),np.float32)
            b=np.zeros((300,360,4),np.float32)
            c=np.ones((300,360),np.uint8)
            out=crop_heldout(d,a,b,c)
            self.assertEqual(out[-2],"full")
            self.assertEqual(out[-1],"legacy_full_region_wald")
            self.assertIs(out[0],a)

    def test_manifest_pixel_scale_must_match(self):
        with TemporaryDirectory() as d:
            self.create_cache(d)
            p=Path(d)/"roi.json"
            payload=json.loads(p.read_text())
            payload["test_bbox_10m"]=[72,108,217,252]
            p.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,"exactly 3x"):
                read_roi(d)


    def test_uafl_loader_filters_every_intersecting_training_tile(self):
        forbidden=(18,30,78,90)
        with TemporaryDirectory() as d:
            train=Path(d)/"train"
            train.mkdir()
            for name,arr in (
                ("gt",np.zeros((99,120,242),np.float32)),
                ("lr_hsi",np.zeros((33,40,242),np.float32)),
                ("hr_msi",np.zeros((99,120,4),np.float32)),
                ("valid_mask",np.ones((99,120),np.uint8)),
            ):
                np.save(train/(name+".npy"),arr)
            (train/"meta.json").write_text(json.dumps({
                "forbidden_bbox_30m":list(forbidden),
                "msi_source":"real_Sentinel_2_Wald_30m",
            }))
            ds=WaldDataset(d,"train",train_patch=24,train_stride=6,
                           eval_patch=48,min_valid_fraction=.8)
            self.assertGreater(len(ds),10)
            for y,x,ph,pw in ds.tiles:
                self.assertFalse(
                    y < forbidden[2] and y+ph > forbidden[0]
                    and x < forbidden[3] and x+pw > forbidden[1]
                )


if __name__=="__main__":
    unittest.main()
