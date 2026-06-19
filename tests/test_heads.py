import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import heads
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

    def test_cascade_output_initialization(self):
        head = heads.CascadePointerHead(8, 3, pointer_size=4, max_span_len=4)
        self.assert_output_linear_initialized(head.start_classifier)
        self.assert_output_linear_initialized(head.start_query)
        self.assert_output_linear_initialized(head.end_key)


class CascadePointerHeadTest(unittest.TestCase):
    def test_uses_banded_end_logits_and_masks_invalid_offsets(self):
        head = heads.CascadePointerHead(4, 2, pointer_size=2, max_span_len=4)
        features = torch.randn(1, 3, 4)
        mask = torch.tensor([[True, True, False]])
        labels = {
            "start_labels": torch.tensor([[0, 0, -100]]),
            "end_labels": torch.full((1, 3), -100),
        }

        output = head(features, labels, mask)

        self.assertEqual(output["logits"]["start"].shape, (1, 3, 3))
        self.assertEqual(output["logits"]["end"].shape, (1, 3, 4))
        self.assertTrue(torch.isfinite(output["loss"]))
        self.assertTrue(torch.all(output["logits"]["end"][0, 1, 1:] < -1e20))

    def test_decode_uses_start_type_and_conditional_end_pointer(self):
        head = heads.CascadePointerHead(4, 2, pointer_size=2, max_span_len=4)
        start_logits = torch.zeros(1, 4, 3)
        start_logits[..., 0] = 1.0
        start_logits[0, 1, 2] = 5.0
        end_logits = torch.full((1, 4, 4), -10.0)
        end_logits[0, 1, 1] = 3.0

        decoded = head.decode(
            {"start": start_logits, "end": end_logits},
            torch.ones(1, 4, dtype=torch.bool),
            {0: "name", 1: "email"},
            ["abcd"],
        )

        self.assertEqual(
            decoded,
            [[{"text": "bc", "type": "email", "start": 1, "end": 3}]],
        )


if __name__ == "__main__":
    unittest.main()
