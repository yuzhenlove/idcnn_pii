import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from heads import CRFHead, EfficientGlobalPointerHead, SoftmaxHead


class OutputInitializationTest(unittest.TestCase):
    def assert_output_linear_initialized(self, layer):
        self.assertTrue(torch.equal(layer.bias, torch.full_like(layer.bias, 0.01)))
        bound = (6.0 / (layer.in_features + layer.out_features)) ** 0.5
        self.assertLessEqual(layer.weight.abs().max().item(), bound)

    def test_softmax_output_initialization(self):
        self.assert_output_linear_initialized(SoftmaxHead(8, 5).classifier)

    def test_crf_output_initialization(self):
        self.assert_output_linear_initialized(CRFHead(8, 5).classifier)

    def test_egp_output_initialization(self):
        head = EfficientGlobalPointerHead(8, 3, head_size=4)
        self.assert_output_linear_initialized(head.qk_proj)
        self.assert_output_linear_initialized(head.bias_proj)


if __name__ == "__main__":
    unittest.main()
