import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from scripts.evaluate_nas_all_candidates import (
    collect_candidates,
    parse_gpu_list,
    read_completed,
)


class EvaluateNasAllCandidatesTest(unittest.TestCase):
    def test_parses_gpu_list(self):
        self.assertEqual(parse_gpu_list("0,1,2,3,4"), ("0", "1", "2", "3", "4"))
        with self.assertRaises(ValueError):
            parse_gpu_list("0,1,0")

    def test_collects_unique_candidates_from_search_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            experiment_dir = root / "experiment_3"
            candidate_id = "0-0-0-3-0-0-1-0-0-2-0-0"
            candidate_dir = experiment_dir / "candidates" / candidate_id
            candidate_dir.mkdir(parents=True)
            (candidate_dir / "best.pt").touch()
            state = {
                "all_results": {
                    candidate_id: {
                        "candidate_id": candidate_id,
                        "individual": [0, 0, 0, 3, 2, 3, 1, 0, 0, 2, 0, 0],
                        "dev_f1": 0.9,
                        "flops": 100,
                    }
                }
            }
            (experiment_dir / "search_state.json").write_text(
                json.dumps(state),
                encoding="utf-8",
            )

            rows = collect_candidates(root, [3])

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["candidate_id"], candidate_id)

    def test_completed_results_are_separate_from_pending(self):
        candidates = [
            {
                "experiment": 3,
                "candidate_id": "a",
            },
            {
                "experiment": 3,
                "candidate_id": "b",
            },
        ]
        with tempfile.TemporaryDirectory() as directory:
            output_dir = Path(directory)
            path = output_dir / "experiment_3" / "a.json"
            path.parent.mkdir()
            path.write_text(
                json.dumps({"experiment": 3, "candidate_id": "a"}),
                encoding="utf-8",
            )

            completed, pending = read_completed(candidates, output_dir)

        self.assertEqual([row["candidate_id"] for row in completed], ["a"])
        self.assertEqual([row["candidate_id"] for row in pending], ["b"])


if __name__ == "__main__":
    unittest.main()
