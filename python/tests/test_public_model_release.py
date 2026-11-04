"""CPU acceptance checks for the bundled, non-detector public model release."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import torch

from argus_research.experimental_cnn import (
    AHEM_E2E_CHECKPOINT_SHA256, RecoveredAhemShadow, TemporalEnsemble,
)

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT = ROOT / "models/challengers/ahem_e2e_experimental_step3808.pt"


class PublicModelReleaseTests(unittest.TestCase):
    def test_sanitized_weights_contract(self):
        payload = torch.load(ARTIFACT, map_location="cpu", weights_only=True)
        self.assertEqual(set(payload), {
            "schema", "config", "step", "seed", "source_checkpoint_sha256",
            "security_detector", "publication_evidence", "promotion_eligible",
            "evaluation_status", "state_dict",
        })
        self.assertEqual(hashlib.sha256(ARTIFACT.read_bytes()).hexdigest(), AHEM_E2E_CHECKPOINT_SHA256)
        self.assertEqual(len(payload["state_dict"]), 59)
        self.assertTrue(all(t.dtype == torch.float32 and torch.isfinite(t).all() for t in payload["state_dict"].values()))
        manifest = json.loads(ARTIFACT.with_suffix(".pt.manifest.json").read_text())
        self.assertEqual(manifest["artifact_sha256"], AHEM_E2E_CHECKPOINT_SHA256)
        self.assertFalse(manifest["includes_optimizer_or_rng_state"])

    def test_tampered_artifact_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.pt"
            path.write_bytes(b"not trusted checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                RecoveredAhemShadow(path)

    def test_recovered_shadow_cpu_inference(self):
        torch.set_num_threads(1)
        shadow = RecoveredAhemShadow(ARTIFACT)
        features = torch.rand((160, 12), generator=torch.Generator().manual_seed(7))
        rows = [{"opaque_flow_id": "0", "source_capture_ordinal": str(i)} for i in range(160)]
        output = shadow.run(features, rows)
        self.assertEqual(output["state"], "forecast_only")
        self.assertEqual(output["forecasts_scored"], 1)
        self.assertFalse(output["security_detector"])
        self.assertFalse(output["routing_eligible"])
        self.assertEqual(len(output["predicted_features"]), 2)
        self.assertTrue(torch.isfinite(torch.tensor(output["predicted_features"])).all())
        self.assertLess(max(output["selected_source_ordinals"]), 158)

    def test_runtime_loads_bundled_v61_and_opt_in_shadow(self):
        ensemble = TemporalEnsemble(ROOT / "models", device="cpu", enable_ahem_e2e_shadow=True)
        self.assertIsNotNone(ensemble.ahem_e2e_shadow, ensemble.rejected)
        self.assertIsNotNone(ensemble.ahem_v61, ensemble.rejected)
        self.assertIsNotNone(ensemble.ahem_v61_onnx, ensemble.rejected)

    def test_frozen_v61_onnx_parity(self):
        from argus_research.export_ahem_v61 import AhemV61OnnxWrapper, verify_parity
        from argus_research.ahem_v61_frozen import AhemV6StateTransitionNextPacket

        torch.set_num_threads(1)
        model = AhemV6StateTransitionNextPacket()
        payload = torch.load(ROOT / "models/ahem_v61_forecast_only.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(payload["state_dict"], strict=True)
        result = verify_parity(AhemV61OnnxWrapper(model.eval()), ROOT / "models/ahem_v61.onnx")
        self.assertTrue(result["verified"])
        self.assertLessEqual(result["max_abs_error"], 1e-4)
        print("v6.1 ONNX parity max_abs_error:", result["max_abs_error"])


if __name__ == "__main__":
    unittest.main()
