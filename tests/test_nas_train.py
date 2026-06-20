import sys
import tempfile
import unittest
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nas_train import (
    apply_archive_initialization,
    build_nas_model,
    candidate_data_paths,
    count_nas_flops,
)


CFG = {
    "data": {
        "processed_dir": "data/processed",
        "train_path": "data/processed/train.jsonl",
        "dev_path": "data/processed/dev.jsonl",
        "test_path": "data/processed/test.jsonl",
    },
    "model": {
        "input_dropout": 0.35,
        "hidden_dropout": 0.15,
        "cascade_max_span_len": 64,
        "cascade_pointer_size": 64,
    },
    "train": {
        "epochs": 100,
        "batch_size": 128,
        "max_len": 512,
        "lr": 0.0005,
        "beta1": 0.9,
        "beta2": 0.9,
        "epsilon": 1e-6,
        "weight_decay": 0.0,
        "grad_clip": 5.0,
        "token_dropout": 0.15,
        "early_stop_patience": 20,
    },
}


class NasTrainTest(unittest.TestCase):
    def test_builds_decoded_nas_model_with_cascade_head(self):
        individual = (1, 0, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2)

        model = build_nas_model(CFG, vocab_size=30, entity_type_num=6, individual=individual, experiment=3)

        self.assertEqual(model.encoder.channels, 128)
        self.assertEqual(model.encoder.ratio, 0.5)
        self.assertEqual(model.encoder.cell_num, 3)
        self.assertEqual(model.head.start_classifier.out_features, 7)

    def test_search_candidate_paths_exclude_test_data(self):
        paths = candidate_data_paths(CFG, ROOT)

        self.assertEqual(set(paths), {"train", "dev", "processed_dir"})
        self.assertNotIn("test", paths)

    def test_flops_use_fixed_length_forward_without_decode(self):
        individual = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        model = build_nas_model(CFG, vocab_size=30, entity_type_num=6, individual=individual, experiment=1)

        flops_128 = count_nas_flops(model, sequence_length=128)
        flops_64 = count_nas_flops(model, sequence_length=64)

        self.assertGreater(flops_128, 0)
        self.assertGreater(flops_128, flops_64)

    def test_archive_initialization_loads_embedding_and_first_compatible_op(self):
        individual = (0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        source = build_nas_model(CFG, 10, 3, individual, experiment=1)
        target = build_nas_model(CFG, 10, 3, individual, experiment=1)
        with torch.no_grad():
            for parameter in source.parameters():
                parameter.fill_(3.0)
            for parameter in target.parameters():
                parameter.zero_()

        with tempfile.TemporaryDirectory() as directory:
            checkpoint_path = Path(directory) / "best.pt"
            torch.save(
                {
                    "model_state_dict": source.state_dict(),
                    "individual": list(individual),
                    "experiment": 1,
                },
                checkpoint_path,
            )
            transferred = apply_archive_initialization(
                target,
                checkpoint_path,
                individual,
                individual,
                CFG,
                vocab_size=10,
                entity_type_num=3,
                experiment=1,
            )

        self.assertEqual(transferred, "op1")
        self.assertTrue(torch.all(target.encoder.embedding.weight == 3.0))
        self.assertTrue(torch.all(target.encoder.cell.operations[0].weight == 3.0))
        self.assertTrue(torch.all(target.encoder.cell.operations[1].weight == 0.0))


if __name__ == "__main__":
    unittest.main()
