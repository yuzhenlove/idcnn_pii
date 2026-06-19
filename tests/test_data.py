import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import collate_batch


class CascadeLabelTest(unittest.TestCase):
    def make_item(self, length, entity):
        return {
            "id": 1,
            "text": "字" * length,
            "input_ids": [1] * length,
            "label_ids": [0] * length,
            "entities": [entity],
        }

    def test_length_64_entity_produces_start_type_and_end_offset(self):
        item = self.make_item(
            64,
            {"text": "字" * 64, "type": "device_identifier", "start": 0, "end": 64},
        )

        batch = collate_batch(
            [item],
            entity2id={"device_identifier": 2},
            cascade_max_span_len=64,
        )

        labels = batch["cascade_labels"]
        self.assertEqual(labels["start_labels"][0, 0].item(), 3)
        self.assertEqual(labels["end_labels"][0, 0].item(), 63)

    def test_length_65_entity_is_not_used_as_cascade_target(self):
        item = self.make_item(
            65,
            {"text": "字" * 65, "type": "authentication", "start": 0, "end": 65},
        )

        batch = collate_batch(
            [item],
            entity2id={"authentication": 0},
            cascade_max_span_len=64,
        )

        labels = batch["cascade_labels"]
        self.assertEqual(labels["start_labels"][0, 0].item(), 0)
        self.assertEqual(labels["end_labels"][0, 0].item(), -100)


if __name__ == "__main__":
    unittest.main()
