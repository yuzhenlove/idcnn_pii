import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import train


class TrainConfigurationTest(unittest.TestCase):
    def test_build_model_uses_configured_idcnn_parameters(self):
        cfg = {
            "model": {
                "embedding_dim": 100,
                "hidden_size": 300,
                "input_dropout": 0.35,
                "hidden_dropout": 0.15,
                "dilations": [1, 2, 1],
                "kernel_size": 3,
            }
        }

        model = train.build_model(cfg, vocab_size=50, output_size=7, num_blocks=2, head="softmax")

        self.assertEqual(model.encoder.embedding.embedding_dim, 100)
        self.assertEqual(model.encoder.initial_conv.out_channels, 300)
        self.assertEqual(model.encoder.input_dropout.p, 0.35)
        self.assertEqual(model.encoder.hidden_dropout.p, 0.15)
        self.assertEqual([layer.dilation[0] for layer in model.encoder.layers], [1, 2, 1])
        self.assertEqual(model.encoder.num_blocks, 2)

    def test_token_dropout_replaces_only_active_tokens(self):
        input_ids = torch.tensor([[2, 3, 0], [4, 0, 0]])
        mask = input_ids.ne(0)

        dropped = train.apply_token_dropout(input_ids, mask, unk_id=1, probability=1.0)

        self.assertTrue(torch.equal(dropped, torch.tensor([[1, 1, 0], [1, 0, 0]])))
        self.assertTrue(torch.equal(input_ids, torch.tensor([[2, 3, 0], [4, 0, 0]])))


if __name__ == "__main__":
    unittest.main()
