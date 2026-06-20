import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.profile_onnx_complexity import OnnxExportWrapper, export_onnx, latency_stats, make_session
from train import build_model
from utils import load_yaml


class ProfileOnnxComplexityTest(unittest.TestCase):
    def test_export_wrapper_returns_named_tensor_outputs(self):
        cfg = load_yaml(ROOT / "configs.yaml")
        input_ids = torch.ones(1, 64, dtype=torch.long)
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        for head, output_size, expected_outputs in [
            ("softmax", 5, 1),
            ("crf", 5, 1),
            ("egp", 3, 1),
            ("cascade", 3, 2),
        ]:
            with self.subTest(head=head):
                model = build_model(cfg, 20, output_size, 1, head).eval()
                outputs = OnnxExportWrapper(model, head)(input_ids, mask)
                if expected_outputs == 1:
                    self.assertIsInstance(outputs, torch.Tensor)
                else:
                    self.assertEqual(len(outputs), expected_outputs)

    def test_latency_stats_uses_milliseconds(self):
        stats = latency_stats([0.001, 0.002, 0.003])

        self.assertAlmostEqual(stats["mean_ms"], 2.0)
        self.assertAlmostEqual(stats["p50_ms"], 2.0)
        self.assertAlmostEqual(stats["p95_ms"], float(np.percentile([1.0, 2.0, 3.0], 95)))

    def test_export_keeps_model_in_eval_mode(self):
        cfg = load_yaml(ROOT / "configs.yaml")
        model = build_model(cfg, 20, 5, 1, "softmax").eval()
        input_ids = torch.ones(1, 64, dtype=torch.long)
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        with tempfile.TemporaryDirectory() as temp_dir:
            export_onnx(model, "softmax", input_ids, mask, Path(temp_dir) / "model.onnx")

        self.assertFalse(model.training)

    def test_exported_egp_model_loads_in_onnx_runtime(self):
        cfg = load_yaml(ROOT / "configs.yaml")
        model = build_model(cfg, 20, 3, 1, "egp").eval()
        input_ids = torch.ones(1, 64, dtype=torch.long)
        mask = torch.ones_like(input_ids, dtype=torch.bool)

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "model.onnx"
            export_onnx(model, "egp", input_ids, mask, path)
            session = make_session(path)
            actual = session.run(None, {"input_ids": input_ids.numpy(), "mask": mask.numpy()})[0]

        with torch.inference_mode():
            expected = OnnxExportWrapper(model, "egp")(input_ids, mask).numpy()

        self.assertEqual(len(session.get_outputs()), 1)
        np.testing.assert_allclose(expected, actual, rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    unittest.main()
