import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import plot_results
import summarize_results


class ReportingTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.outputs_dir = self.root / "outputs"
        self.logs_dir = self.root / "logs"
        for head_index, head in enumerate(["softmax", "crf", "egp", "cascade"]):
            for num_blocks in [1, 2, 3, 4]:
                for seed in [42, 43, 44]:
                    score = 0.65 + 0.06 * head_index + 0.01 * num_blocks + 0.001 * (seed - 42)
                    metrics = {
                        "run_id": f"{head}_b{num_blocks}_seed{seed}",
                        "head": head,
                        "num_blocks": num_blocks,
                        "seed": seed,
                        "best_epoch": 50,
                        "dev": {
                            "micro": {
                                "precision": score,
                                "recall": score,
                                "f1": score,
                            }
                        },
                        "test": {
                            "micro": {
                                "precision": score,
                                "recall": score,
                                "f1": score,
                            }
                        },
                    }
                    run_dir = self.outputs_dir / metrics["run_id"]
                    run_dir.mkdir(parents=True)
                    (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_generate_report_filters_three_heads(self):
        report_dir = summarize_results.generate_report(
            self.outputs_dir,
            ["softmax", "crf", "egp"],
            "baseline",
            self.logs_dir,
        )

        summary = summarize_results.pd.read_csv(report_dir / "summary.csv")
        mean_std = summarize_results.pd.read_csv(report_dir / "summary_mean_std.csv")

        self.assertEqual(len(summary), 36)
        self.assertEqual(len(mean_std), 12)
        self.assertEqual(set(summary["head"]), {"softmax", "crf", "egp"})
        self.assertTrue((self.logs_dir / "experiments_baseline.csv").exists())

    def test_generate_report_includes_four_heads(self):
        report_dir = summarize_results.generate_report(
            self.outputs_dir,
            ["softmax", "crf", "egp", "cascade"],
            "all_heads",
            self.logs_dir,
        )

        summary = summarize_results.pd.read_csv(report_dir / "summary.csv")
        mean_std = summarize_results.pd.read_csv(report_dir / "summary_mean_std.csv")

        self.assertEqual(len(summary), 48)
        self.assertEqual(len(mean_std), 16)
        self.assertEqual(set(summary["head"]), {"softmax", "crf", "egp", "cascade"})

    def test_generate_figures_supports_four_heads(self):
        summarize_results.generate_report(
            self.outputs_dir,
            ["softmax", "crf", "egp", "cascade"],
            "all_heads",
            self.logs_dir,
        )

        figures_dir = plot_results.generate_figures(
            self.outputs_dir,
            ["softmax", "crf", "egp", "cascade"],
            "all_heads",
        )

        figure_files = list(figures_dir.glob("*.*"))
        self.assertEqual(len(figure_files), 21)
        self.assertEqual({path.suffix for path in figure_files}, {".png", ".pdf", ".svg"})


if __name__ == "__main__":
    unittest.main()
