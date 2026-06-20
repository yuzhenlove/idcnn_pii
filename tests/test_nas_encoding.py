import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nas_encoding import (
    crossover,
    decode_individual,
    hamming_distance,
    individual_key,
    mutate,
    normalize_individual,
    random_individual,
)


class NasEncodingTest(unittest.TestCase):
    def test_decodes_twelve_gene_individual(self):
        decoded = decode_individual((1, 2, 3, 0, 1, 2, 1, 2, 3, 2, 0, 1), experiment=3)

        self.assertEqual(decoded["C"], 128)
        self.assertEqual(decoded["ratio"], 2.0)
        self.assertEqual(decoded["cell_num"], 4)
        self.assertEqual(decoded["ops"][0], {"type": "conv", "kernel_size": 5, "dilation": 4})
        self.assertEqual(decoded["ops"][1], {"type": "dwconv", "kernel_size": 7, "dilation": 8})
        self.assertEqual(decoded["ops"][2], {"type": "sepconv", "kernel_size": 3, "dilation": 2})

    def test_identity_genes_are_normalized_before_key_and_distance(self):
        left = (0, 0, 0, 3, 2, 3, 0, 0, 0, 0, 0, 0)
        right = (0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0)

        self.assertEqual(normalize_individual(left, 3), right)
        self.assertEqual(individual_key(left, 3), individual_key(right, 3))
        self.assertEqual(hamming_distance(left, right, 3), 0)

    def test_experiment_one_fixes_convolutions_kernel_and_dilations(self):
        raw = (2, 1, 3, 3, 2, 3, 1, 2, 3, 2, 2, 3)

        normalized = normalize_individual(raw, 1)

        self.assertEqual(normalized, (2, 1, 3, 0, 0, 0, 0, 0, 1, 0, 0, 2))

    def test_experiment_two_fixes_operation_type_only(self):
        raw = (2, 1, 3, 3, 2, 3, 1, 2, 3, 2, 2, 3)

        normalized = normalize_individual(raw, 2)

        self.assertEqual(normalized, (2, 1, 3, 0, 2, 3, 0, 2, 3, 0, 2, 3))

    def test_random_crossover_and_mutation_keep_legal_canonical_values(self):
        rng = random.Random(7)
        parent_a = random_individual(3, rng)
        parent_b = random_individual(3, rng)
        child_a, child_b = crossover(parent_a, parent_b, 3, rng, probability=1.0)
        mutated = mutate(child_a, 3, rng, gene_probability=1.0)

        for individual in (parent_a, parent_b, child_a, child_b, mutated):
            self.assertEqual(len(individual), 12)
            self.assertEqual(individual, normalize_individual(individual, 3))
        self.assertNotEqual(mutated, child_a)


if __name__ == "__main__":
    unittest.main()
