"""CPU acceptance checks for the bundled, non-detector public model release."""

from pathlib import Path
import tempfile
import unittest

import torch

from argus_research.experimental_cnn import RecoveredAhemShadow, TemporalEnsemble

ROOT = Path(__file__).resolve().parents[2]
RETIRED_SHADOW = ROOT / "models/challengers/ahem_e2e_experimental_step3808.pt"


class PublicModelReleaseTests(unittest.TestCase):
    def test_retired_unvalidated_shadow_not_bundled(self):
        self.assertFalse(RETIRED_SHADOW.exists())
        self.assertFalse(RETIRED_SHADOW.with_suffix(".pt.manifest.json").exists())

    def test_tampered_artifact_rejected_before_deserialization(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.pt"
            path.write_bytes(b"not trusted checkpoint bytes")
            with self.assertRaisesRegex(ValueError, "SHA-256 differs"):
                RecoveredAhemShadow(path)

    def test_runtime_loads_bundled_v61(self):
        ensemble = TemporalEnsemble(ROOT / "models", device="cpu", enable_ahem_e2e_shadow=False)
        self.assertIsNone(ensemble.ahem_e2e_shadow)
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
