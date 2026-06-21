import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

METRIC_FIELDS = [
    "individual",
    "architecture",
    "dev_f1",
    "test_precision",
    "test_recall",
    "test_f1",
    "parameters",
    "flops",
    "onnx_forward_mean_ms",
    "decode_entities_mean_ms",
    "complete_mean_latency_ms",
    "complete_p50_latency_ms",
    "complete_p95_latency_ms",
    "throughput_samples_per_second",
    "checkpoint",
    "onnx_file",
]


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def index_unique(rows: list[dict], label: str) -> dict[tuple[int, str], dict]:
    indexed = {}
    for row in rows:
        key = (int(row["experiment"]), row["candidate_id"])
        if key in indexed:
            raise ValueError(f"{label} contains duplicate key: {key}")
        indexed[key] = row
    return indexed


def merge_metrics(test_rows: list[dict], latency_rows: list[dict]) -> dict:
    tests = index_unique(test_rows, "Test CSV")
    latency = index_unique(latency_rows, "latency CSV")
    missing_latency = sorted(set(tests) - set(latency))
    missing_test = sorted(set(latency) - set(tests))
    if missing_latency or missing_test:
        raise ValueError(
            f"candidate key mismatch: missing_latency={len(missing_latency)} "
            f"missing_test={len(missing_test)}"
        )
    merged = {}
    for key, test in tests.items():
        speed = latency[key]
        if test["flops"] != speed["flops"]:
            raise ValueError(f"FLOPs mismatch for {key}")
        merged[key] = {
            "experiment": key[0],
            "candidate_id": key[1],
            "individual": test["individual"],
            "architecture": speed["architecture"],
            "dev_f1": test["dev_f1"],
            "test_precision": test["test_precision"],
            "test_recall": test["test_recall"],
            "test_f1": test["test_f1"],
            "parameters": speed["parameters"],
            "flops": speed["flops"],
            "onnx_forward_mean_ms": speed["onnx_forward_mean_ms"],
            "decode_entities_mean_ms": speed["decode_entities_mean_ms"],
            "complete_mean_latency_ms": speed["complete_mean_latency_ms"],
            "complete_p50_latency_ms": speed["complete_p50_latency_ms"],
            "complete_p95_latency_ms": speed["complete_p95_latency_ms"],
            "throughput_samples_per_second": speed["throughput_samples_per_second"],
            "checkpoint": test["checkpoint"],
            "onnx_file": speed["onnx_file"],
        }
    return merged


def build_pareto_table(pareto_path: Path, experiment: int, metrics: dict) -> list[dict]:
    pareto = json.loads(pareto_path.read_text(encoding="utf-8"))
    rows = []
    for position, candidate in enumerate(pareto, 1):
        key = (experiment, candidate["candidate_id"])
        if key not in metrics:
            raise ValueError(f"Pareto candidate missing metrics: {key}")
        rows.append({"pareto_position": position, **metrics[key]})
    return rows


def build_generation_table(generations_dir: Path, experiment: int, metrics: dict) -> list[dict]:
    rows = []
    files = sorted(generations_dir.glob("generation_*.json"))
    if not files:
        raise FileNotFoundError(f"no generation files found in {generations_dir}")
    for path in files:
        generation_data = json.loads(path.read_text(encoding="utf-8"))
        generation = int(generation_data["generation"])
        candidates = (
            generation_data["population"]
            if generation == 0
            else generation_data["offspring"]
        )
        for slot, candidate in enumerate(candidates, 1):
            key = (experiment, candidate["candidate_id"])
            if key not in metrics:
                raise ValueError(f"generation candidate missing metrics: {key}")
            rows.append(
                {
                    "generation": generation,
                    "generation_slot": slot,
                    **metrics[key],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict], leading_fields: list[str]) -> None:
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[*leading_fields, "experiment", "candidate_id", *METRIC_FIELDS],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument(
        "--test-csv",
        type=Path,
        default=ROOT / "outputs" / "reports" / "nas_all_candidates_test.csv",
    )
    parser.add_argument(
        "--latency-csv",
        type=Path,
        default=ROOT
        / "outputs"
        / "reports"
        / "nas_all_candidates_latency_128_onnx_xeon8463b.csv",
    )
    parser.add_argument(
        "--table1-output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "table1_pareto_metrics.csv",
    )
    parser.add_argument(
        "--table2-output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "table2_all_generations_metrics.csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = merge_metrics(read_csv(args.test_csv), read_csv(args.latency_csv))
    experiment_dir = ROOT / "outputs" / "nas" / f"experiment_{args.experiment}"
    table1 = build_pareto_table(
        experiment_dir / "pareto.json",
        args.experiment,
        metrics,
    )
    table2 = build_generation_table(
        experiment_dir / "generations",
        args.experiment,
        metrics,
    )
    write_csv(args.table1_output, table1, ["pareto_position"])
    write_csv(args.table2_output, table2, ["generation", "generation_slot"])
    print(f"wrote table1 rows={len(table1)} to {args.table1_output}")
    print(f"wrote table2 rows={len(table2)} to {args.table2_output}")


if __name__ == "__main__":
    main()
