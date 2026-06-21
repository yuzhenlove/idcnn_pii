import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.profile_nas_onnx_latency import (
    TEST_TEXT,
    collect_candidates,
    display_path,
    onnx_result_exists,
    validate_rows,
)


class ProfileNasOnnxLatencyTest(unittest.TestCase):
    def test_fixed_text_has_128_character_tokens(self):
        self.assertEqual(len(TEST_TEXT), 128)

    def test_collect_candidates_normalizes_and_deduplicates_records(self):
        individual = [0, 0, 0, 3, 2, 3, 1, 0, 0, 2, 0, 0]
        normalized_id = "0-0-0-3-0-0-1-0-0-2-0-0"
        with tempfile.TemporaryDirectory() as directory:
            nas_root = Path(directory)
            experiment_dir = nas_root / "experiment_3"
            candidate_dir = experiment_dir / "candidates" / normalized_id
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "best.pt").touch()
            record = {
                "individual": individual,
                "candidate_id": normalized_id,
                "dev_f1": 0.9,
                "flops": 123,
                "checkpoint": "/stale/server/path/best.pt",
            }
            (experiment_dir / "search_state.json").write_text(
                json.dumps({"all_results": {normalized_id: record}}),
                encoding="utf-8",
            )
            (candidate_dir / "candidate.json").write_text(
                json.dumps(record),
                encoding="utf-8",
            )

            rows, stats = collect_candidates(nas_root, [3])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], normalized_id)
        self.assertEqual(rows[0]["checkpoint"], candidate_dir / "best.pt")
        self.assertEqual(stats["state_records"], 1)
        self.assertEqual(stats["candidate_result_files"], 1)
        self.assertEqual(stats["discovered_checkpoints"], 1)
        self.assertEqual(stats["unique_candidates"], 1)
        self.assertEqual(stats["missing_checkpoints"], 0)

    def test_display_path_keeps_external_paths_absolute(self):
        self.assertTrue(Path(display_path(Path("/tmp/external.pt"))).is_absolute())

    def test_onnx_result_requires_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.onnx"
            self.assertFalse(onnx_result_exists({"onnx_file": str(path)}))
            path.touch()
            self.assertTrue(onnx_result_exists({"onnx_file": str(path)}))

    def test_validate_rows_rejects_duplicate_candidates(self):
        row = {
            "experiment": 3,
            "candidate_id": "candidate",
            "input_length": 128,
            "batch_size": 1,
            "cpu_threads": 1,
            "provider": "CPUExecutionProvider",
        }
        with self.assertRaises(ValueError):
            validate_rows([row, dict(row)])


if __name__ == "__main__":
    unittest.main()
