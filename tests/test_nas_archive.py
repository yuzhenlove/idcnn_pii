import sys
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model_nas_idcnn import NASIDCNNEncoder, NASIDCNNForNER
from nas_archive import select_weight_source, transfer_partial_weights, update_archive


def candidate(individual, f1, flops, checkpoint="checkpoint.pt"):
    return {
        "individual": list(individual),
        "dev_f1": f1,
        "flops": flops,
        "checkpoint": checkpoint,
    }


class ArchiveUpdateTest(unittest.TestCase):
    def test_update_respects_capacity_and_removes_normalized_duplicates(self):
        rows = []
        for index in range(40):
            individual = (
                index % 4,
                (index // 4) % 3,
                (index // 12) % 4,
                0,
                index % 3,
                index % 4,
                1,
                (index + 1) % 3,
                (index + 1) % 4,
                2,
                (index + 2) % 3,
                (index + 2) % 4,
            )
            rows.append(candidate(individual, 0.7 + index / 1000, 1000 - index))
        duplicate_identity_a = candidate((0, 0, 0, 3, 2, 3, 0, 0, 0, 0, 0, 0), 0.95, 100)
        duplicate_identity_b = candidate((0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0), 0.90, 120)

        archive = update_archive([], rows + [duplicate_identity_a, duplicate_identity_b], 3, capacity=30, top_k=3)

        self.assertLessEqual(len(archive), 30)
        identity_rows = [row for row in archive if row["individual"][3] == 3 and row["individual"][:3] == [0, 0, 0]]
        self.assertEqual(len(identity_rows), 1)
        self.assertEqual(identity_rows[0]["dev_f1"], 0.95)

    def test_selects_same_width_ratio_then_distance_and_f1(self):
        target = (1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        archive = [
            candidate((0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2), 0.99, 100, "wrong-width.pt"),
            candidate((1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2), 0.80, 100, "near-low.pt"),
            candidate((1, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2), 0.90, 110, "near-high.pt"),
        ]

        source = select_weight_source(target, archive, experiment=1)

        self.assertEqual(source["checkpoint"], "near-high.pt")


class DummyHead(torch.nn.Module):
    def forward(self, features, labels=None, mask=None):
        return {"loss": features.mean(), "logits": features}


def make_model(operations):
    return NASIDCNNForNER(
        NASIDCNNEncoder(
            vocab_size=10,
            channels=4,
            ratio=1.0,
            cell_num=1,
            operations=operations,
            input_dropout=0.0,
            hidden_dropout=0.0,
        ),
        DummyHead(),
    )


class WeightTransferTest(unittest.TestCase):
    def test_transfers_embedding_and_only_first_compatible_convolution(self):
        operations = [
            {"type": "conv", "kernel_size": 3, "dilation": 1},
            {"type": "conv", "kernel_size": 3, "dilation": 2},
            {"type": "conv", "kernel_size": 3, "dilation": 4},
        ]
        source = make_model(operations)
        target = make_model(operations)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(5.0)
            for parameter in target.parameters():
                parameter.zero_()
        individual = (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)

        transferred = transfer_partial_weights(target, source, individual, individual, experiment=1)

        self.assertEqual(transferred, "op1")
        self.assertTrue(torch.all(target.encoder.embedding.weight == 5.0))
        self.assertTrue(torch.all(target.encoder.cell.operations[0].weight == 5.0))
        self.assertTrue(torch.all(target.encoder.cell.operations[1].weight == 0.0))


if __name__ == "__main__":
    unittest.main()
