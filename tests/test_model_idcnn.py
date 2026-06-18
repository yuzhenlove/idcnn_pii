import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_idcnn import IDCNNEncoder


class IDCNNEncoderTest(unittest.TestCase):
    def test_default_architecture_matches_author_idcnn_block(self):
        encoder = IDCNNEncoder(
            vocab_size=50,
            input_dropout=0.0,
            hidden_dropout=0.0,
        )

        self.assertEqual(encoder.embedding.embedding_dim, 100)
        self.assertEqual(encoder.initial_conv.in_channels, 100)
        self.assertEqual(encoder.initial_conv.out_channels, 300)
        self.assertEqual(encoder.initial_conv.kernel_size, (3,))
        self.assertEqual(encoder.initial_conv.dilation, (1,))
        self.assertEqual([layer.dilation[0] for layer in encoder.layers], [1, 2, 1])
        self.assertTrue(all(layer.in_channels == 300 for layer in encoder.layers))
        self.assertTrue(all(layer.out_channels == 300 for layer in encoder.layers))

    def test_block_convolutions_use_identity_initialization(self):
        encoder = IDCNNEncoder(
            vocab_size=10,
            embedding_dim=4,
            hidden_size=4,
            input_dropout=0.0,
            hidden_dropout=0.0,
            dilations=[1, 2, 1],
        )

        expected = torch.zeros_like(encoder.layers[0].weight)
        expected[:, :, 1] = torch.eye(4)
        for layer in encoder.layers:
            self.assertTrue(torch.equal(layer.weight, expected))
            self.assertTrue(torch.equal(layer.bias, torch.zeros_like(layer.bias)))
        self.assertTrue(torch.allclose(encoder.initial_conv.bias, torch.full_like(encoder.initial_conv.bias, 0.01)))

    def test_block_does_not_add_residual_input(self):
        encoder = IDCNNEncoder(
            vocab_size=3,
            embedding_dim=1,
            hidden_size=1,
            input_dropout=0.0,
            hidden_dropout=0.0,
            dilations=[1],
            num_blocks=1,
        )
        with torch.no_grad():
            encoder.embedding.weight.fill_(1.0)
            encoder.initial_conv.weight.zero_()
            encoder.initial_conv.bias.fill_(1.0)
            encoder.layers[0].weight.zero_()
            encoder.layers[0].bias.zero_()

        input_ids = torch.ones(1, 3, dtype=torch.long)
        output = encoder(input_ids, input_ids.ne(0))

        self.assertTrue(torch.equal(output, torch.zeros_like(output)))

    def test_outputs_are_invariant_to_right_padding(self):
        torch.manual_seed(0)
        seq = torch.arange(1, 13).unsqueeze(0) % 49 + 1
        padded = torch.cat([seq, torch.zeros(1, 28, dtype=torch.long)], dim=1)

        for num_blocks in [1, 2, 3, 4]:
            with self.subTest(num_blocks=num_blocks):
                encoder = IDCNNEncoder(
                    vocab_size=50,
                    embedding_dim=8,
                    hidden_size=8,
                    input_dropout=0.0,
                    hidden_dropout=0.0,
                    num_blocks=num_blocks,
                )
                encoder.eval()
                with torch.no_grad():
                    short_out = encoder(seq, seq.ne(0))
                    padded_out = encoder(padded, padded.ne(0))[:, : seq.size(1)]

                self.assertTrue(torch.allclose(short_out, padded_out, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
