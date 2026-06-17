import argparse
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    args = parser.parse_args()

    outputs_dir = ROOT / args.outputs_dir
    rows = []
    for path in sorted(outputs_dir.glob("*/metrics.json")):
        with open(path, "r", encoding="utf-8") as f:
            rows.append(flatten(json.load(f)))
    df = pd.DataFrame(rows)
    outputs_dir.mkdir(parents=True, exist_ok=True)
    (ROOT / "logs").mkdir(parents=True, exist_ok=True)
    df.to_csv(outputs_dir / "summary.csv", index=False)
    df.to_csv(ROOT / "logs" / "experiments.csv", index=False)
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
    mean_std.to_csv(outputs_dir / "summary_mean_std.csv", index=False)
    print(
        f"wrote {len(df)} rows to {outputs_dir / 'summary.csv'}, "
        f"{ROOT / 'logs' / 'experiments.csv'}, and {outputs_dir / 'summary_mean_std.csv'}"
    )


if __name__ == "__main__":
    main()
