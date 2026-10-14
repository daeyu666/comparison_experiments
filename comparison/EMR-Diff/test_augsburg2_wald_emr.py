"""CPU structural/provenance checks for EMR Augsburg center-heldout Wald v1."""
import importlib.util
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
    HSI_BANDS, MSI_BANDS, STATE_BANDS, PROTOCOL, WALD_ARCHITECTURE_ID,
    build_diffusion, build_model, masked_l1, pack_batch, verify_checkpoint,
    training_step, predict,
)
from arch.BAFUnet import BAFUNet
from EMRDiff import Edge
_infer_spec = importlib.util.spec_from_file_location(
    "emr_infer_augsburg2_wald", HERE / "infer_augsburg2_wald.py"
)
_infer_module = importlib.util.module_from_spec(_infer_spec)
_infer_spec.loader.exec_module(_infer_module)
parse_infer_args = _infer_module.parse_args


class WaldEMRCenterHoldoutChecks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_batch_two_dynamic_io_training_and_reverse(self):
        model = build_model(64, "cpu")
        self.assertIsInstance(model, BAFUNet)
        self.assertEqual((model.in_channels, model.model_channels, model.out_channels),
                         (246, 64, 246))
        self.assertFalse(any(key.startswith("core.") or key.startswith("state_head.")
                             for key in model.state_dict()))
        sample = {
            "gt": torch.rand(2, 242, 24, 24),
            "lr_hsi": torch.rand(2, 242, 8, 8),
            "hr_msi": torch.rand(2, 4, 24, 24),
            "valid_mask": torch.ones(2, 1, 24, 24),
        }
        gt, lq, msi, mask, _ = pack_batch(sample, "cpu", (np.ones(4), np.zeros(4)))
        diffusion, edge = build_diffusion("cpu"), Edge()
        loss = training_step(model, diffusion, edge, gt, lq, msi, mask)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(model.residual_heads["64"].weight.grad).all())
        prediction = predict(model, diffusion, edge, lq, msi)
        self.assertEqual(tuple(prediction.shape), (2, 242, 32, 32))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_new_wald_checkpoint_save_and_strict_reload(self):
        import json
        from tempfile import TemporaryDirectory
        from types import SimpleNamespace
        spec = importlib.util.spec_from_file_location(
            "emr_train_augsburg2_wald", HERE / "train_augsburg2_wald.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        save_state, load_state = module.save_state, module.load_state

        with TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "train").mkdir()
            (root / "train" / "meta.json").write_text(json.dumps({"training_area": "synthetic_fixture"}))
            args = SimpleNamespace(wald_root=str(root), model_width=64, monitor="ref_sam", batch_size=2)
            model = build_model(64, "cpu")
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
            geometry = dict(split_protocol_id=CENTER_PROTOCOL,
                            test_bbox_30m=[24, 36, 72, 84],
                            forbidden_bbox_30m=[18, 30, 78, 90])
            checkpoint = root / "new.pth.tar"
            save_state(checkpoint, model=model, optimizer=optimizer, epoch=0,
                       best=1.0, args=args, sha="fixture", sigma=1.2, **geometry)
            restored = build_model(64, "cpu")
            state = load_state(checkpoint, restored, sha="fixture", sigma=1.2,
                               device="cpu", monitor="ref_sam", **geometry)
            self.assertEqual(state["architecture_id"], WALD_ARCHITECTURE_ID)
            self.assertEqual((state["backbone_width"], state["state_channels"]), (64, 246))
            self.assertEqual(state["args"]["batch_size"], 2)
            torch.testing.assert_close(restored.residual_heads["64"].weight,
                                       model.residual_heads["64"].weight)

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
                forbidden_bbox_30m=[18, 30, 78, 90],
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
            "forbidden_bbox_30m": [18, 30, 78, 90],
            "radiometry_sha256": "abc",
            "wald_sigma": 1.2,
            "model_width": 64,
            "architecture_id": WALD_ARCHITECTURE_ID,
            "backbone_width": 64,
            "state_channels": 246,
            "monitor": "ref_sam",
        }
        verify_checkpoint(
            state, radiometry_sha="abc", sigma=1.2,
            split_protocol_id=CENTER_PROTOCOL,
            test_bbox_30m=[24, 36, 72, 84],
            forbidden_bbox_30m=[18, 30, 78, 90],
            monitor="ref_sam", width=64,
        )
        for field, value in [("architecture_id", None), ("state_channels", 64),
                             ("backbone_width", 34),
                             ("architecture_id", "fixed64_dynamic_io_v2")]:
            old = dict(state)
            old[field] = value
            with self.assertRaisesRegex(ValueError, "Incompatible EMR Wald architecture"):
                verify_checkpoint(
                    old, radiometry_sha="abc", sigma=1.2,
                    split_protocol_id=CENTER_PROTOCOL,
                    test_bbox_30m=[24, 36, 72, 84],
                    forbidden_bbox_30m=[18, 30, 78, 90],
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
