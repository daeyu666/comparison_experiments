"""Lightweight structural tests; do not require Augsburg input files or GPU.

Run from repository root:
    python -m unittest discover -s comparison/EMR-Diff -p "test_augsburg2_wald_emr.py"
Set EMR_WALD_TEST_MODEL=1 to include the slower CPU backbone shape test.
"""
import os
import unittest

import torch

from wald_emr_common import (
    HSI_BANDS, MSI_BANDS, STATE_BANDS, build_diffusion, build_model,
    masked_l1, pack_batch,
)


class WaldEMRChecks(unittest.TestCase):
    def test_fixed_observed_dimensions(self):
        self.assertEqual((HSI_BANDS, MSI_BANDS, STATE_BANDS), (242,4,246))
        diffusion = build_diffusion("cpu")
        self.assertEqual(diffusion.num_diffusion_timesteps, 5)
        self.assertEqual(diffusion.band_dim, 242)

    def test_x3_72_crop_pads_network_only(self):
        sample = {
            "gt": torch.ones((1,242,72,72)),
            "lr_hsi": torch.ones((1,242,24,24)),
            "hr_msi": torch.full((1,4,72,72),2.0),
            "valid_mask": torch.ones((1,1,72,72)),
        }
        gt,lq,msi,mask,hw = pack_batch(
            sample, "cpu",
            (torch.tensor([0.5,0.5,0.5,0.5]), torch.zeros(4)),
        )
        self.assertEqual(hw,(72,72))
        self.assertEqual(tuple(gt.shape),(1,242,80,80))
        self.assertEqual(tuple(lq.shape),(1,242,80,80))
        self.assertEqual(tuple(msi.shape),(1,4,80,80))
        self.assertAlmostEqual(float(msi[0,0,0,0]),1.0)
        self.assertEqual(int(mask.sum()),72*72)
        self.assertEqual(float(mask[:,:,72:,:].sum()),0.0)
        self.assertEqual(float(mask[:,:,:,72:].sum()),0.0)

    def test_mask_rejects_padded_prediction_error(self):
        target = torch.zeros((1,246,80,80))
        prediction = target.clone()
        prediction[:,:,72:,:] = 5.
        prediction[:,:,:,72:] = 5.
        mask = torch.zeros((1,1,80,80))
        mask[:,:,:72,:72] = 1.
        self.assertEqual(float(masked_l1(prediction,target,mask)),0.)

    @unittest.skipUnless(os.getenv("EMR_WALD_TEST_MODEL")=="1",
                         "Set EMR_WALD_TEST_MODEL=1 for CPU model-forward shape test")
    def test_network_output_width(self):
        model=build_model(32, "cpu").eval()
        x=torch.zeros((1,246,48,48))
        m=torch.zeros((1,4,48,48))
        lq=torch.zeros((1,242,48,48))
        with torch.no_grad():
            out, features=model(x,m,lq,torch.zeros((1,),dtype=torch.long))
        self.assertEqual(tuple(out.shape),(1,246,48,48))
        self.assertTrue(len(features)>6)
        self.assertEqual(features[2].shape[1],246)


if __name__=="__main__":
    unittest.main()
