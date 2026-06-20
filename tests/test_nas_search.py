import io
import json
import os
import random
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from scripts.run_nas_search import (
    assign_candidates_to_gpus,
    build_candidate_command,
    evaluate_population,
    estimate_candidate_cost,
    estimate_training_seconds,
    generate_offspring,
    generate_unique_initial_population,
    load_cached_results,
    parse_args,
    parse_gpu_list,
    run_parallel_jobs,
    run_search,
    search_budget,
    write_pareto_outputs,
)
from nas_encoding import individual_key, normalize_individual, random_individual
from nsga2 import assign_rank_and_crowding


class NasSearchTest(unittest.TestCase):
    def test_structural_cost_increases_with_width_and_cell_count(self):
        small = (0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        wide = (3, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        repeated = (0, 0, 3, 0, 0, 0, 0, 0, 1, 0, 0, 2)

        self.assertGreater(
            estimate_candidate_cost(wide, experiment=1),
            estimate_candidate_cost(small, experiment=1),
        )
        self.assertGreater(
            estimate_candidate_cost(repeated, experiment=1),
            estimate_candidate_cost(small, experiment=1),
        )

    def test_training_time_estimate_prefers_same_width_ratio_history(self):
        target = (3, 2, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2)
        history = [
            {
                "individual": [3, 2, 0, 0, 0, 0, 0, 0, 1, 0, 0, 2],
                "train_seconds": 1200.0,
            },
            {
                "individual": [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 2],
                "train_seconds": 100.0,
            },
        ]

        estimate = estimate_training_seconds(target, history, experiment=1)

        self.assertEqual(estimate, 1200.0)

    def test_balances_longest_candidates_across_gpu_capacity(self):
        candidates = [
            ("a", 10.0),
            ("b", 9.0),
            ("c", 8.0),
            ("d", 7.0),
            ("e", 6.0),
            ("f", 5.0),
        ]

        assignments = assign_candidates_to_gpus(
            candidates,
            gpus=("0", "1", "2"),
            workers_per_gpu=2,
        )

        loads = {"0": 0.0, "1": 0.0, "2": 0.0}
        counts = {"0": 0, "1": 0, "2": 0}
        for candidate, cost, gpu_id in assignments:
            loads[gpu_id] += cost
            counts[gpu_id] += 1

        self.assertEqual(loads, {"0": 15.0, "1": 15.0, "2": 15.0})
        self.assertEqual(counts, {"0": 2, "1": 2, "2": 2})

    def test_parses_comma_separated_gpu_list(self):
        self.assertEqual(parse_gpu_list("0,1,2,3,4"), ("0", "1", "2", "3", "4"))

    def test_rejects_empty_or_duplicate_gpu_ids(self):
        with self.assertRaises(ValueError):
            parse_gpu_list("")
        with self.assertRaises(ValueError):
            parse_gpu_list("0,1,0")

    def test_parallel_jobs_assign_two_round_robin_slots_per_gpu(self):
        class CompletedProcess:
            def poll(self):
                return 0

        jobs = []
        with tempfile.TemporaryDirectory() as directory:
            for index in range(10):
                jobs.append((["python", str(index)], Path(directory) / str(index)))

            with patch(
                "scripts.run_nas_search.subprocess.Popen",
                side_effect=lambda *args, **kwargs: CompletedProcess(),
            ) as popen:
                with redirect_stdout(io.StringIO()):
                    run_parallel_jobs(
                        jobs,
                        workers=10,
                        status_interval=60,
                        gpus=("0", "1", "2", "3", "4"),
                        workers_per_gpu=2,
                    )

        assigned = [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list]
        self.assertEqual(assigned, ["0", "1", "2", "3", "4", "0", "1", "2", "3", "4"])
        for call in popen.call_args_list:
            self.assertEqual(call.kwargs["env"]["CUDA_DEVICE_ORDER"], "PCI_BUS_ID")
            self.assertEqual(call.kwargs["env"].get("PATH"), os.environ.get("PATH"))

    def test_parallel_jobs_honor_preassigned_gpu(self):
        class CompletedProcess:
            def poll(self):
                return 0

        with tempfile.TemporaryDirectory() as directory:
            jobs = [
                (["python", "slow"], Path(directory) / "slow", "4"),
                (["python", "fast"], Path(directory) / "fast", "0"),
            ]
            with patch(
                "scripts.run_nas_search.subprocess.Popen",
                side_effect=lambda *args, **kwargs: CompletedProcess(),
            ) as popen:
                with redirect_stdout(io.StringIO()):
                    run_parallel_jobs(
                        jobs,
                        workers=2,
                        status_interval=60,
                        gpus=("0", "1", "2", "3", "4"),
                        workers_per_gpu=1,
                    )

        assigned = [call.kwargs["env"]["CUDA_VISIBLE_DEVICES"] for call in popen.call_args_list]
        self.assertEqual(assigned, ["4", "0"])

    def test_population_evaluation_preassigns_balanced_gpu_jobs(self):
        individuals = [
            (index % 4, index % 3, index % 4, 0, 0, 0, 0, 0, 1, 0, 0, 2)
            for index in range(6)
        ]
        completed = {
            individual_key(individual, 1): {
                "individual": list(normalize_individual(individual, 1)),
                "dev_f1": 0.8,
                "flops": 100,
                "checkpoint": "best.pt",
            }
            for individual in individuals
        }

        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "scripts.run_nas_search.load_cached_results",
                side_effect=[({}, individuals), (completed, [])],
            ):
                with patch("scripts.run_nas_search.run_parallel_jobs") as run_jobs:
                    with redirect_stdout(io.StringIO()):
                        evaluate_population(
                            individuals,
                            experiment=1,
                            candidates_dir=Path(directory),
                            archive=[],
                            workers=6,
                            status_interval=60,
                            epochs=1,
                            batch_size=4,
                            cpu=False,
                            gpus=("0", "1", "2"),
                            workers_per_gpu=2,
                        )

        scheduled_jobs = run_jobs.call_args.args[0]
        assigned = [job[2] for job in scheduled_jobs]
        self.assertEqual({gpu_id: assigned.count(gpu_id) for gpu_id in set(assigned)}, {
            "0": 2,
            "1": 2,
            "2": 2,
        })

    def test_search_cli_accepts_gpu_slots(self):
        argv = [
            "run_nas_search.py",
            "--experiment",
            "3",
            "--gpus",
            "0,1,2,3,4",
            "--workers-per-gpu",
            "2",
        ]

        with patch("sys.argv", argv):
            args = parse_args()

        self.assertEqual(args.gpus, ("0", "1", "2", "3", "4"))
        self.assertEqual(args.workers_per_gpu, 2)

    def test_dry_run_prints_balanced_gpu_assignments(self):
        args = SimpleNamespace(
            search_seed=42,
            population_size=10,
            dry_run=True,
            epochs=1,
            batch_size=4,
            cpu=False,
            gpus=("0", "1", "2", "3", "4"),
            workers_per_gpu=2,
            force=False,
        )

        with tempfile.TemporaryDirectory() as directory:
            output = io.StringIO()
            with patch("scripts.run_nas_search.ROOT", Path(directory)):
                with redirect_stdout(output):
                    run_search(args, experiment=3)

        assignments = [
            line.split()[0]
            for line in output.getvalue().splitlines()
            if line.startswith("CUDA_VISIBLE_DEVICES=")
        ]
        rng = random.Random(42)
        individuals = generate_unique_initial_population(10, 3, rng)
        expected = [
            f"CUDA_VISIBLE_DEVICES={gpu_id}"
            for _individual, _cost, gpu_id in assign_candidates_to_gpus(
                [
                    (individual, estimate_candidate_cost(individual, 3))
                    for individual in individuals
                ],
                ("0", "1", "2", "3", "4"),
                2,
            )
        ]
        self.assertEqual(assignments, expected)

    def test_balancer_queues_more_candidates_than_concurrent_slots(self):
        assignments = assign_candidates_to_gpus(
            [(str(index), float(20 - index)) for index in range(9)],
            gpus=("0", "1"),
            workers_per_gpu=2,
        )

        counts = [
            sum(gpu_id == item for _candidate, _cost, gpu_id in assignments)
            for item in ("0", "1")
        ]
        self.assertEqual(sorted(counts), [4, 5])

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
