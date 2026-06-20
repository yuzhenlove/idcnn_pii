import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_nas_pareto import pareto_test_path


class ParetoTestEvaluationTest(unittest.TestCase):
    def test_explicit_final_evaluator_uses_test_path(self):
        cfg = {"data": {"test_path": "data/processed/test.jsonl"}}

        path = pareto_test_path(cfg, ROOT)

        self.assertEqual(path, ROOT / "data/processed/test.jsonl")


if __name__ == "__main__":
    unittest.main()
