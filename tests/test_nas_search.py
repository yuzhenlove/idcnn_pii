import json
import random
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_nas_search import (
    build_candidate_command,
    generate_offspring,
    load_cached_results,
    search_budget,
    write_pareto_outputs,
)
from nas_encoding import individual_key, normalize_individual, random_individual
from nsga2 import assign_rank_and_crowding


class NasSearchTest(unittest.TestCase):
    def test_budget_excludes_initial_population_from_generation_count(self):
        self.assertEqual(search_budget(population_size=10, generations=5), 60)
        self.assertEqual(search_budget(population_size=10, generations=50), 510)

    def test_candidate_command_includes_archive_source_when_available(self):
        individual = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        source = {
            "individual": list(individual),
            "checkpoint": "/tmp/source/best.pt",
        }

        command = build_candidate_command(
            individual,
            experiment=1,
            output_dir=Path("/tmp/target"),
            source=source,
            epochs=2,
            batch_size=4,
            cpu=True,
        )

        self.assertIn("--source-checkpoint", command)
        self.assertIn("/tmp/source/best.pt", command)
        self.assertIn("--source-individual", command)
        self.assertIn("--cpu", command)

    def test_offspring_are_canonical_and_match_population_size(self):
        rng = random.Random(42)
        parents = []
        for index in range(10):
            row = {
                "individual": list(random_individual(3, rng)),
                "dev_f1": 0.8 + index / 100,
                "flops": 1000 - index,
            }
            parents.append(row)
        ranked = assign_rank_and_crowding(parents)

        offspring = generate_offspring(ranked, 10, experiment=3, rng=rng)

        self.assertEqual(len(offspring), 10)
        self.assertTrue(
            all(tuple(row) == normalize_individual(row, 3) for row in offspring)
        )

    def test_cached_normalized_duplicate_is_loaded_once(self):
        identity_a = (0, 0, 0, 3, 2, 3, 0, 0, 0, 0, 0, 0)
        identity_b = (0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 0)
        with tempfile.TemporaryDirectory() as directory:
            candidate_dir = Path(directory) / individual_key(identity_a, 3)
            candidate_dir.mkdir()
            result = {
                "individual": list(normalize_individual(identity_a, 3)),
                "dev_f1": 0.9,
                "flops": 100,
                "checkpoint": str(candidate_dir / "best.pt"),
            }
            (candidate_dir / "candidate.json").write_text(json.dumps(result), encoding="utf-8")

            cached, missing = load_cached_results(
                [identity_a, identity_b],
                Path(directory),
                experiment=3,
            )

        self.assertEqual(len(cached), 1)
        self.assertEqual(missing, [])

    def test_writes_only_global_non_dominated_candidates(self):
        rows = [
            {"individual": [0] * 12, "candidate_id": "a", "dev_f1": 0.8, "flops": 80, "checkpoint": "a.pt"},
            {"individual": [1] * 12, "candidate_id": "b", "dev_f1": 0.9, "flops": 100, "checkpoint": "b.pt"},
            {"individual": [2] * 12, "candidate_id": "c", "dev_f1": 0.7, "flops": 120, "checkpoint": "c.pt"},
        ]
        with tempfile.TemporaryDirectory() as directory:
            pareto = write_pareto_outputs(rows, Path(directory))
            csv_text = (Path(directory) / "pareto.csv").read_text(encoding="utf-8")

        self.assertEqual({row["candidate_id"] for row in pareto}, {"a", "b"})
        self.assertNotIn("c,", csv_text)


if __name__ == "__main__":
    unittest.main()
