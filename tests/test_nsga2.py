import random
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nsga2 import assign_rank_and_crowding, binary_tournament, dominates, environmental_selection


def candidate(name, f1, flops):
    return {"name": name, "dev_f1": f1, "flops": flops}


class NSGA2Test(unittest.TestCase):
    def test_dominance_maximizes_f1_and_minimizes_flops(self):
        strong = candidate("strong", 0.90, 100)
        weak = candidate("weak", 0.80, 120)
        tradeoff = candidate("tradeoff", 0.95, 140)

        self.assertTrue(dominates(strong, weak))
        self.assertFalse(dominates(strong, tradeoff))
        self.assertFalse(dominates(tradeoff, strong))

    def test_assigns_fronts_and_infinite_boundary_crowding(self):
        ranked = assign_rank_and_crowding(
            [
                candidate("fast", 0.80, 80),
                candidate("balanced", 0.90, 100),
                candidate("accurate", 0.95, 140),
                candidate("dominated", 0.70, 160),
            ]
        )
        by_name = {row["name"]: row for row in ranked}

        self.assertEqual([by_name[name]["rank"] for name in ("fast", "balanced", "accurate")], [0, 0, 0])
        self.assertEqual(by_name["dominated"]["rank"], 1)
        self.assertEqual(by_name["fast"]["crowding_distance"], float("inf"))
        self.assertEqual(by_name["accurate"]["crowding_distance"], float("inf"))

    def test_environmental_selection_prefers_rank_then_crowding(self):
        selected = environmental_selection(
            [
                candidate("fast", 0.80, 80),
                candidate("middle", 0.85, 100),
                candidate("accurate", 0.90, 140),
                candidate("dominated", 0.70, 160),
            ],
            population_size=2,
        )

        self.assertEqual({row["name"] for row in selected}, {"fast", "accurate"})

    def test_binary_tournament_prefers_lower_rank(self):
        population = [
            {**candidate("best", 0.9, 100), "rank": 0, "crowding_distance": 1.0},
            {**candidate("worse", 0.8, 120), "rank": 1, "crowding_distance": float("inf")},
        ]

        winner = binary_tournament(population, random.Random(1), indices=(0, 1))

        self.assertEqual(winner["name"], "best")


if __name__ == "__main__":
    unittest.main()
