import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.build_nas_result_tables import (
    build_generation_table,
    build_pareto_table,
    merge_metrics,
)


def metric(candidate_id: str) -> dict:
    return {
        "experiment": 3,
        "candidate_id": candidate_id,
        "individual": "[]",
        "architecture": "{}",
        "dev_f1": "0.8",
        "test_precision": "0.7",
        "test_recall": "0.6",
        "test_f1": "0.65",
        "parameters": "10",
        "flops": "20",
        "onnx_forward_mean_ms": "1.0",
        "decode_entities_mean_ms": "0.1",
        "complete_mean_latency_ms": "1.1",
        "complete_p50_latency_ms": "1.1",
        "complete_p95_latency_ms": "1.2",
        "throughput_samples_per_second": "909",
        "checkpoint": "best.pt",
        "onnx_file": "model.onnx",
    }


class BuildNasResultTablesTest(unittest.TestCase):
    def test_merges_test_and_latency_by_candidate_key(self):
        test = metric("a")
        latency = metric("a")

        merged = merge_metrics([test], [latency])

        self.assertIn((3, "a"), merged)
        self.assertEqual(merged[(3, "a")]["test_f1"], "0.65")
        self.assertEqual(merged[(3, "a")]["complete_mean_latency_ms"], "1.1")

    def test_pareto_table_preserves_pareto_order(self):
        metrics = {(3, "a"): metric("a"), (3, "b"): metric("b")}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pareto.json"
            path.write_text(
                json.dumps([{"candidate_id": "b"}, {"candidate_id": "a"}]),
                encoding="utf-8",
            )

            rows = build_pareto_table(path, 3, metrics)

        self.assertEqual([row["candidate_id"] for row in rows], ["b", "a"])
        self.assertEqual([row["pareto_position"] for row in rows], [1, 2])

    def test_generation_table_keeps_duplicate_slots(self):
        metrics = {(3, "a"): metric("a"), (3, "b"): metric("b")}
        with tempfile.TemporaryDirectory() as directory:
            generations = Path(directory)
            (generations / "generation_000.json").write_text(
                json.dumps(
                    {
                        "generation": 0,
                        "population": [
                            {"candidate_id": "a"},
                            {"candidate_id": "a"},
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (generations / "generation_001.json").write_text(
                json.dumps(
                    {
                        "generation": 1,
                        "offspring": [
                            {"candidate_id": "b"},
                            {"candidate_id": "a"},
                        ],
                    }
                ),
                encoding="utf-8",
            )

            rows = build_generation_table(generations, 3, metrics)

        self.assertEqual(len(rows), 4)
        self.assertEqual([row["candidate_id"] for row in rows], ["a", "a", "b", "a"])
        self.assertEqual([row["generation"] for row in rows], [0, 0, 1, 1])


if __name__ == "__main__":
    unittest.main()
