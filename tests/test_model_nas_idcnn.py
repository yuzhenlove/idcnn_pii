import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_nas_idcnn import NASIDCNNEncoder, NASIDCNNForNER, SearchCell, build_search_operation


class SearchOperationTest(unittest.TestCase):
    def test_operations_produce_expected_channels(self):
        x = torch.randn(2, 8, 16)
        cases = [
            ("conv", 12),
            ("dwconv", 8),
            ("sepconv", 12),
            ("identity", 8),
        ]

        for operation_type, expected_channels in cases:
            with self.subTest(operation_type=operation_type):
                operation, output_channels = build_search_operation(operation_type, 8, 12, 3, 2)
                self.assertEqual(operation(x).shape, (2, expected_channels, 16))
                self.assertEqual(output_channels, expected_channels)

    def test_depthwise_convolution_uses_one_group_per_channel(self):
        operation, _ = build_search_operation("dwconv", 8, 12, 5, 4)

        self.assertEqual(operation.groups, 8)
        self.assertEqual(operation.in_channels, operation.out_channels)


class SearchCellTest(unittest.TestCase):
    def test_tracks_mixed_operation_channels_and_restores_backbone_width(self):
        cell = SearchCell(
            channels=8,
            bottleneck_channels=4,
            operations=[
                {"type": "conv", "kernel_size": 3, "dilation": 1},
                {"type": "dwconv", "kernel_size": 5, "dilation": 2},
                {"type": "sepconv", "kernel_size": 3, "dilation": 4},
            ],
        )

        output = cell(torch.randn(2, 8, 12))

        self.assertEqual(output.shape, (2, 8, 12))
        self.assertEqual(cell.operation_channels, [(8, 4), (4, 4), (4, 4)])
        self.assertEqual(cell.final_conv.in_channels, 4)
        self.assertEqual(cell.final_conv.out_channels, 8)

    def test_identity_skips_activation(self):
        cell = SearchCell(
            channels=2,
            bottleneck_channels=2,
            operations=[
                {"type": "identity", "kernel_size": 3, "dilation": 1},
                {"type": "identity", "kernel_size": 3, "dilation": 1},
                {"type": "identity", "kernel_size": 3, "dilation": 1},
            ],
        )
        with torch.no_grad():
            cell.final_conv.weight.zero_()
            cell.final_conv.bias.fill_(-1.0)

        output = cell(torch.ones(1, 2, 3))

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))
        self.assertTrue(all(isinstance(activation, nn.Identity) for activation in cell.activations))


class CountingHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def forward(self, features, labels=None, mask=None):
        self.calls += 1
        return {"loss": features.mean() if labels is not None else None, "logits": features}


class NASIDCNNEncoderTest(unittest.TestCase):
    def test_reuses_one_cell_instance_for_all_repetitions(self):
        encoder = NASIDCNNEncoder(
            vocab_size=20,
            channels=8,
            ratio=0.5,
            cell_num=3,
            operations=[{"type": "conv", "kernel_size": 3, "dilation": 1}] * 3,
            input_dropout=0.0,
            hidden_dropout=0.0,
        )
        calls = []
        handle = encoder.cell.register_forward_hook(lambda *_: calls.append(1))

        output = encoder(torch.tensor([[1, 2, 3]]), torch.tensor([[True, True, True]]))
        handle.remove()

        self.assertEqual(output.shape, (1, 3, 8))
        self.assertEqual(len(calls), 3)
        self.assertEqual(sum(name == "cell" for name, _ in encoder.named_modules()), 1)

    def test_model_sends_only_last_cell_output_to_head(self):
        encoder = NASIDCNNEncoder(
            vocab_size=20,
            channels=4,
            ratio=1.0,
            cell_num=4,
            operations=[{"type": "identity", "kernel_size": 3, "dilation": 1}] * 3,
            input_dropout=0.0,
            hidden_dropout=0.0,
        )
        head = CountingHead()
        model = NASIDCNNForNER(encoder, head)
        input_ids = torch.tensor([[1, 2, 3]])

        output = model(input_ids, {"start_labels": torch.zeros(1, 3)}, input_ids.ne(0))

        self.assertEqual(head.calls, 1)
        self.assertIsNotNone(output["loss"])


if __name__ == "__main__":
    unittest.main()
