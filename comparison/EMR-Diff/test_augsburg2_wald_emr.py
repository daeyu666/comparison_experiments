"""CPU structural/provenance checks for EMR Augsburg center-heldout Wald v1."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
UAFL = HERE.parent / "UAFL"
if str(UAFL) not in sys.path:
    sys.path.insert(0, str(UAFL))

from augsburg2_wald_center_roi import PROTOCOL as CENTER_PROTOCOL, crop_heldout
from wald_emr_common import (
    HSI_BANDS, MSI_BANDS, STATE_BANDS, PROTOCOL,
    build_diffusion, build_model, masked_l1, pack_batch, verify_checkpoint,
)
from infer_augsburg2_wald import parse_args as parse_infer_args


class WaldEMRCenterHoldoutChecks(unittest.TestCase):
    def test_fixed_observed_dimensions(self):
        self.assertEqual((HSI_BANDS, MSI_BANDS, STATE_BANDS), (242, 4, 246))
        diffusion = build_diffusion("cpu")
        self.assertEqual(diffusion.num_diffusion_timesteps, 5)
        self.assertEqual(diffusion.band_dim, 242)

    def test_train_24_crop_pads_network_only_to_32(self):
        sample = {
            "gt": torch.ones((1, 242, 24, 24)),
            "lr_hsi": torch.ones((1, 242, 8, 8)),
            "hr_msi": torch.full((1, 4, 24, 24), 2.0),
            "valid_mask": torch.ones((1, 1, 24, 24)),
        }
        gt, lq, msi, mask, hw = pack_batch(
            sample, "cpu",
            (torch.tensor([0.5, 0.5, 0.5, 0.5]), torch.zeros(4)),
        )
        self.assertEqual(hw, (24, 24))
        self.assertEqual(tuple(gt.shape), (1, 242, 32, 32))
        self.assertEqual(tuple(lq.shape), (1, 242, 32, 32))
        self.assertEqual(tuple(msi.shape), (1, 4, 32, 32))
        self.assertAlmostEqual(float(msi[0, 0, 0, 0]), 1.0)
        self.assertEqual(int(mask.sum()), 24 * 24)
        self.assertEqual(float(mask[:, :, 24:, :].sum()), 0.0)
        self.assertEqual(float(mask[:, :, :, 24:].sum()), 0.0)

    def test_eval_48_crop_pads_network_only_to_48(self):
        sample = {
            "gt": torch.ones((1, 242, 48, 48)),
            "lr_hsi": torch.ones((1, 242, 16, 16)),
            "hr_msi": torch.ones((1, 4, 48, 48)),
            "valid_mask": torch.ones((1, 1, 48, 48)),
        }
        gt, lq, msi, mask, hw = pack_batch(
            sample, "cpu", (np.ones(4, np.float32), np.zeros(4, np.float32))
        )
        self.assertEqual(hw, (48, 48))
        self.assertEqual(tuple(gt.shape), (1, 242, 48, 48))
        self.assertEqual(tuple(lq.shape), (1, 242, 48, 48))
        self.assertEqual(int(mask.sum()), 48 * 48)

    def test_mask_rejects_padded_prediction_error(self):
        target = torch.zeros((1, 246, 32, 32))
        prediction = target.clone()
        prediction[:, :, 24:, :] = 5.0
        prediction[:, :, :, 24:] = 5.0
        mask = torch.zeros((1, 1, 32, 32))
        mask[:, :, :24, :24] = 1.0
        self.assertEqual(float(masked_l1(prediction, target, mask)), 0.0)

    def test_legacy_checkpoint_is_rejected(self):
        state = {
            "protocol": PROTOCOL,
            "msi_source": "real_Sentinel_2_Wald_30m",
            "target": "observed_30m_HSI_only",
            "scale_ratio": 3,
            "split_protocol_id": "legacy_full_region_wald",
            "test_bbox_30m": None,
            "radiometry_sha256": "abc",
            "wald_sigma": 1.2,
            "model_width": 64,
            "monitor": "ref_sam",
        }
        with self.assertRaisesRegex(ValueError, "Retrain from scratch"):
            verify_checkpoint(
                state, radiometry_sha="abc", sigma=1.2,
                split_protocol_id=CENTER_PROTOCOL,
                test_bbox_30m=[24, 36, 72, 84],
                monitor="ref_sam", width=64,
            )

    def test_exact_center_checkpoint_is_accepted(self):
        state = {
            "protocol": PROTOCOL,
            "msi_source": "real_Sentinel_2_Wald_30m",
            "target": "observed_30m_HSI_only",
            "scale_ratio": 3,
            "split_protocol_id": CENTER_PROTOCOL,
            "test_bbox_30m": [24, 36, 72, 84],
            "radiometry_sha256": "abc",
            "wald_sigma": 1.2,
            "model_width": 64,
            "monitor": "ref_sam",
        }
        verify_checkpoint(
            state, radiometry_sha="abc", sigma=1.2,
            split_protocol_id=CENTER_PROTOCOL,
            test_bbox_30m=[24, 36, 72, 84],
            monitor="ref_sam", width=64,
        )

    def test_center_inference_defaults_route_to_emr_owned_outputs(self):
        with patch("sys.argv", ["infer_augsburg2_wald.py", "--center_holdout"]):
            args = parse_infer_args()
        self.assertEqual(
            args.wald_root,
            "../S2Diff-MH/data/augsburg2_wald_center_holdout",
        )
        self.assertIn(
            "comparison/EMR-Diff/checkpoints/augsburg2_wald_center_holdout",
            args.checkpoint,
        )
        self.assertEqual(
            args.save_root,
            "./comparison/EMR-Diff/outputs/augsburg2_wald_center_holdout",
        )
        self.assertIn("center_holdout_radiometry", args.radiometry_json)
        self.assertFalse(args.skip_qnr)

    def test_crop_heldout_uses_exact_native_center(self):
        from tempfile import TemporaryDirectory
        import json

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "roi.json").write_text(
                json.dumps({
                    "protocol_id": CENTER_PROTOCOL,
                    "test_bbox_30m": [24, 36, 72, 84],
                    "test_bbox_10m": [72, 108, 216, 252],
                }),
                encoding="utf-8",
            )
            lr = np.zeros((100, 120, 242), np.float32)
            msi = np.zeros((300, 360, 4), np.float32)
            mask = np.ones((300, 360), np.uint8)
            low, high, valid, oy, ox, suffix, protocol = crop_heldout(
                root, lr, msi, mask,
                ckpt_protocol_id=CENTER_PROTOCOL,
                ckpt_bbox_30m=[24, 36, 72, 84],
            )
            self.assertEqual(low.shape, (48, 48, 242))
            self.assertEqual(high.shape, (144, 144, 4))
            self.assertEqual(valid.shape, (144, 144))
            self.assertEqual((oy, ox, suffix, protocol),
                             (72, 108, "heldout", CENTER_PROTOCOL))

    @unittest.skipUnless(
        os.getenv("EMR_WALD_TEST_MODEL") == "1",
        "Set EMR_WALD_TEST_MODEL=1 for the slower CPU backbone shape test",
    )
    def test_network_output_width(self):
        model = build_model(32, "cpu").eval()
        x = torch.zeros((1, 246, 48, 48))
        msi = torch.zeros((1, 4, 48, 48))
        lq = torch.zeros((1, 242, 48, 48))
        with torch.no_grad():
            out, features = model(
                x, msi, lq, torch.zeros((1,), dtype=torch.long)
            )
        self.assertEqual(tuple(out.shape), (1, 246, 48, 48))
        self.assertTrue(len(features) > 6)
        self.assertEqual(features[2].shape[1], 246)


if __name__ == "__main__":
    unittest.main()
