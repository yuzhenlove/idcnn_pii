import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
HEAD_CHOICES = ["softmax", "crf", "egp", "cascade"]


def flatten(metrics: dict) -> dict:
    row = {
        "run_id": metrics["run_id"],
        "head": metrics["head"],
        "num_blocks": metrics["num_blocks"],
        "seed": metrics["seed"],
        "best_epoch": metrics["best_epoch"],
    }
    for split in ["dev", "test"]:
        micro = metrics[split]["micro"]
        row[f"{split}_precision"] = micro["precision"]
        row[f"{split}_recall"] = micro["recall"]
        row[f"{split}_f1"] = micro["f1"]
    return row


def generate_report(
    outputs_dir: str | Path,
    heads: list[str],
    tag: str,
    logs_dir: str | Path,
) -> Path:
    outputs_dir = Path(outputs_dir)
    logs_dir = Path(logs_dir)
    report_dir = outputs_dir / "reports" / tag
    rows = []
    for path in sorted(outputs_dir.glob("*/metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        if metrics["head"] in heads:
            rows.append(flatten(metrics))
    df = pd.DataFrame(rows)
    report_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(report_dir / "summary.csv", index=False)
    df.to_csv(logs_dir / f"experiments_{tag}.csv", index=False)
    if df.empty:
        mean_std = pd.DataFrame()
    else:
        mean_std = (
            df.groupby(["head", "num_blocks"], as_index=False)
            .agg(
                dev_f1_mean=("dev_f1", "mean"),
                dev_f1_std=("dev_f1", "std"),
                test_f1_mean=("test_f1", "mean"),
                test_f1_std=("test_f1", "std"),
                runs=("run_id", "count"),
            )
            .fillna(0.0)
        )
    mean_std.to_csv(report_dir / "summary_mean_std.csv", index=False)
    print(
        f"wrote {len(df)} rows to {report_dir / 'summary.csv'}, "
        f"{logs_dir / f'experiments_{tag}.csv'}, and {report_dir / 'summary_mean_std.csv'}"
    )
    return report_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--heads", nargs="+", choices=HEAD_CHOICES, default=HEAD_CHOICES)
    parser.add_argument("--tag", default="all_heads")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = ROOT / outputs_dir
    generate_report(outputs_dir, args.heads, args.tag, ROOT / "logs")


if __name__ == "__main__":
    main()
