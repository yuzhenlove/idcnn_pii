import sys
import unittest
from pathlib import Path

import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.profile_model_complexity import TEST_TEXT, checkpoint_display_path, count_forward_macs


class ProfileModelComplexityTest(unittest.TestCase):
    def test_fixed_text_has_128_character_tokens(self):
        self.assertEqual(len(TEST_TEXT), 128)

    def test_counts_conv_and_linear_macs(self):
        model = nn.Sequential(
            nn.Conv1d(2, 3, kernel_size=3, padding=1, bias=False),
            nn.Flatten(1),
            nn.Linear(12, 5, bias=False),
        )
        inputs = torch.ones(1, 2, 4)

        macs = count_forward_macs(model, (inputs,))

        self.assertEqual(macs, 1 * 4 * 3 * 2 * 3 + 1 * 12 * 5)

    def test_checkpoint_path_is_relative_to_project_root(self):
        self.assertEqual(
            checkpoint_display_path(Path("outputs/example/best.pt")),
            "outputs/example/best.pt",
        )


if __name__ == "__main__":
    unittest.main()
