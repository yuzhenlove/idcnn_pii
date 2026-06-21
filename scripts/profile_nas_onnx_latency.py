import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from nas_encoding import decode_individual, individual_key, normalize_individual
from nas_train import build_nas_model, count_nas_flops
from scripts.profile_onnx_complexity import (
    OnnxExportWrapper,
    export_onnx,
    make_session,
    measure,
)


TEST_TEXT = (
    "兹授权本公司员工吴雪梅（身份证尾号：6821，工号：ACC-9044）全权处理与迪摩网络实业集团的"
    "对账事宜，包括但不限于签署对账单、确认应收账款等。授权期限至2025年12月31日。公司地址："
    "巢湖市经济开发区趋势网络有限公司办公楼68F。本授权书一式两份。"
)
assert len(TEST_TEXT) == 128

FIELDNAMES = [
    "experiment",
    "candidate_id",
    "individual",
    "architecture",
    "dev_f1",
    "checkpoint",
    "onnx_file",
    "input_length",
    "batch_size",
    "cpu_threads",
    "parameters",
    "macs",
    "flops",
    "onnx_forward_mean_ms",
    "decode_entities_mean_ms",
    "complete_mean_latency_ms",
    "complete_p50_latency_ms",
    "complete_p95_latency_ms",
    "throughput_samples_per_second",
    "provider",
]
ERROR_FIELDNAMES = [
    "experiment",
    "candidate_id",
    "checkpoint",
    "error_type",
    "error_message",
]


def display_path(path: Path, root: Path = ROOT) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def resolve_checkpoint(experiment_dir: Path, candidate_id: str, recorded_path: str | None) -> Path:
    local_path = experiment_dir / "candidates" / candidate_id / "best.pt"
    if local_path.exists():
        return local_path
    if recorded_path:
        path = Path(recorded_path)
        if path.exists():
            return path
    return local_path


def collect_candidates(nas_root: Path, experiments: list[int]) -> tuple[list[dict], dict]:
    candidates = {}
    state_records = 0
    candidate_result_files = 0
    discovered_checkpoints = 0

    for experiment in experiments:
        experiment_dir = nas_root / f"experiment_{experiment}"
        state_path = experiment_dir / "search_state.json"
        records = []
        if state_path.exists():
            state = json.loads(state_path.read_text(encoding="utf-8"))
            state_rows = state.get("all_results", {})
            state_rows = list(state_rows.values()) if isinstance(state_rows, dict) else state_rows
            state_records += len(state_rows)
            records.extend(state_rows)

        result_paths = sorted((experiment_dir / "candidates").glob("*/candidate.json"))
        candidate_result_files += len(result_paths)
        for result_path in result_paths:
            records.append(json.loads(result_path.read_text(encoding="utf-8")))

        discovered_checkpoints += sum(
            1 for _ in (experiment_dir / "candidates").glob("*/best.pt")
        )
        for record in records:
            individual = normalize_individual(record["individual"], experiment)
            candidate_id = individual_key(individual, experiment)
            checkpoint = resolve_checkpoint(
                experiment_dir,
                candidate_id,
                record.get("checkpoint"),
            )
            candidates[(experiment, candidate_id)] = {
                "experiment": experiment,
                "candidate_id": candidate_id,
                "individual": list(individual),
                "architecture": decode_individual(individual, experiment),
                "dev_f1": record.get("dev_f1"),
                "search_flops": record.get("flops"),
                "checkpoint": checkpoint,
            }

    rows = sorted(candidates.values(), key=lambda row: (row["experiment"], row["candidate_id"]))
    missing = [row for row in rows if not row["checkpoint"].is_file()]
    stats = {
        "state_records": state_records,
        "candidate_result_files": candidate_result_files,
        "discovered_checkpoints": discovered_checkpoints,
        "unique_candidates": len(rows),
        "missing_checkpoints": len(missing),
        "per_experiment": {
            experiment: sum(row["experiment"] == experiment for row in rows)
            for experiment in experiments
        },
    }
    if missing:
        sample = ", ".join(
            f"experiment_{row['experiment']}/{row['candidate_id']}" for row in missing[:5]
        )
        raise FileNotFoundError(
            f"{len(missing)} candidate checkpoints are missing; first entries: {sample}"
        )
    return rows, stats


def make_inputs(checkpoint: dict) -> tuple[torch.Tensor, torch.Tensor]:
    char2id = checkpoint["char2id"]
    unk_id = char2id["<UNK>"]
    input_ids = torch.tensor(
        [[char2id.get(char, unk_id) for char in TEST_TEXT]],
        dtype=torch.long,
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    return input_ids, mask


def decode_entities(model, outputs: list[np.ndarray], mask: torch.Tensor, checkpoint: dict):
    logits = {
        "start": torch.from_numpy(outputs[0]),
        "end": torch.from_numpy(outputs[1]),
    }
    id2entity = {index: name for name, index in checkpoint["entity2id"].items()}
    return model.head.decode(logits, mask, id2entity, [TEST_TEXT])


def profile_candidate(
    candidate: dict,
    onnx_root: Path,
    warmup: int,
    iterations: int,
) -> dict:
    checkpoint_path = candidate["checkpoint"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    experiment = candidate["experiment"]
    individual = normalize_individual(checkpoint["individual"], checkpoint["experiment"])
    checkpoint_id = individual_key(individual, checkpoint["experiment"])
    if checkpoint["experiment"] != experiment or checkpoint_id != candidate["candidate_id"]:
        raise ValueError(
            f"checkpoint identity mismatch: expected experiment={experiment} "
            f"candidate={candidate['candidate_id']}, got experiment={checkpoint['experiment']} "
            f"candidate={checkpoint_id}"
        )

    model = build_nas_model(
        checkpoint["config"],
        vocab_size=len(checkpoint["char2id"]),
        entity_type_num=len(checkpoint["entity2id"]),
        individual=individual,
        experiment=experiment,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    input_ids, mask = make_inputs(checkpoint)

    onnx_path = (
        onnx_root
        / f"experiment_{experiment}"
        / f"{candidate['candidate_id']}.onnx"
    )
    export_onnx(model, "cascade", input_ids, mask, onnx_path)
    session = make_session(onnx_path)
    feeds = {"input_ids": input_ids.numpy(), "mask": mask.numpy()}

    with torch.inference_mode():
        torch_outputs = OnnxExportWrapper(model, "cascade")(input_ids, mask)
    onnx_outputs = session.run(None, feeds)
    for expected, actual in zip(torch_outputs, onnx_outputs):
        np.testing.assert_allclose(
            expected.detach().numpy(),
            actual,
            rtol=1e-4,
            atol=1e-4,
        )

    cached_outputs = session.run(None, feeds)
    forward = measure(lambda: session.run(None, feeds), warmup, iterations)
    decoding = measure(
        lambda: decode_entities(model, cached_outputs, mask, checkpoint),
        warmup,
        iterations,
    )

    def complete_inference():
        outputs = session.run(None, feeds)
        decode_entities(model, outputs, mask, checkpoint)

    complete = measure(complete_inference, warmup, iterations)
    flops = count_nas_flops(model, sequence_length=len(TEST_TEXT))
    if candidate["search_flops"] is not None and flops != candidate["search_flops"]:
        raise ValueError(
            f"FLOPs mismatch for {candidate['candidate_id']}: "
            f"search={candidate['search_flops']} rebuilt={flops}. "
            "Check that nas_encoding.py matches the search server."
        )

    return {
        "experiment": experiment,
        "candidate_id": candidate["candidate_id"],
        "individual": json.dumps(list(individual), ensure_ascii=False),
        "architecture": json.dumps(
            decode_individual(individual, experiment),
            ensure_ascii=False,
            sort_keys=True,
        ),
        "dev_f1": candidate["dev_f1"],
        "checkpoint": display_path(checkpoint_path),
        "onnx_file": display_path(onnx_path),
        "input_length": len(TEST_TEXT),
        "batch_size": 1,
        "cpu_threads": 1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "macs": flops // 2,
        "flops": flops,
        "onnx_forward_mean_ms": forward["mean_ms"],
        "decode_entities_mean_ms": decoding["mean_ms"],
        "complete_mean_latency_ms": complete["mean_ms"],
        "complete_p50_latency_ms": complete["p50_ms"],
        "complete_p95_latency_ms": complete["p95_ms"],
        "throughput_samples_per_second": 1000.0 / complete["mean_ms"],
        "provider": "CPUExecutionProvider",
    }


def read_completed_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def validate_rows(rows: list[dict]) -> None:
    keys = [(int(row["experiment"]), row["candidate_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("latency CSV contains duplicate experiment/candidate_id rows")
    for field, expected in [
        ("input_length", "128"),
        ("batch_size", "1"),
        ("cpu_threads", "1"),
        ("provider", "CPUExecutionProvider"),
    ]:
        actual = {str(row[field]) for row in rows}
        if actual and actual != {expected}:
            raise ValueError(f"unexpected {field} values: {sorted(actual)}")


def write_rows(path: Path, rows: list[dict]) -> None:
    validate_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def write_error(path: Path, candidate: dict, error: Exception) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    if path.exists():
        with path.open(encoding="utf-8", newline="") as file:
            rows = list(csv.DictReader(file))
    rows.append(
        {
            "experiment": candidate["experiment"],
            "candidate_id": candidate["candidate_id"],
            "checkpoint": display_path(candidate["checkpoint"]),
            "error_type": type(error).__name__,
            "error_message": str(error),
        }
    )
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=ERROR_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)


def onnx_result_exists(row: dict) -> bool:
    path = Path(row["onnx_file"])
    return path.exists() if path.is_absolute() else (ROOT / path).exists()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--onnx-dir",
        type=Path,
        default=ROOT / "outputs" / "nas_onnx_128",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "nas_all_candidates_latency_128_onnx.csv",
    )
    parser.add_argument(
        "--errors-output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "nas_all_candidates_latency_128_onnx_errors.csv",
    )
    args = parser.parse_args()
    if args.warmup < 0:
        parser.error("--warmup must not be negative")
    if args.iterations < 1:
        parser.error("--iterations must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    experiments = [1, 2, 3] if args.experiment == "all" else [int(args.experiment)]
    candidates, stats = collect_candidates(ROOT / "outputs" / "nas", experiments)

    print(f"search_state records: {stats['state_records']}")
    print(f"candidate.json files: {stats['candidate_result_files']}")
    print(f"discovered checkpoints: {stats['discovered_checkpoints']}")
    print(f"normalized unique candidates: {stats['unique_candidates']}")
    print(f"candidates by experiment: {stats['per_experiment']}")
    print(f"missing checkpoints: {stats['missing_checkpoints']}")

    completed_rows = read_completed_rows(args.output)
    completed = {
        (int(row["experiment"]), row["candidate_id"])
        for row in completed_rows
        if onnx_result_exists(row)
    }
    pending = [
        row
        for row in candidates
        if (row["experiment"], row["candidate_id"]) not in completed
    ]
    if args.limit is not None:
        pending = pending[: args.limit]

    total = len(pending)
    for index, candidate in enumerate(pending, 1):
        try:
            row = profile_candidate(candidate, args.onnx_dir, args.warmup, args.iterations)
        except Exception as error:
            write_error(args.errors_output, candidate, error)
            print(
                f"failed experiment={candidate['experiment']} "
                f"candidate={candidate['candidate_id']}: {error}",
                file=sys.stderr,
                flush=True,
            )
            raise
        completed_rows.append(row)
        completed_rows.sort(key=lambda item: (int(item["experiment"]), item["candidate_id"]))
        write_rows(args.output, completed_rows)
        print(
            f"[{index}/{total}] experiment={row['experiment']} "
            f"candidate={row['candidate_id']} "
            f"forward={row['onnx_forward_mean_ms']:.3f} ms "
            f"complete={row['complete_mean_latency_ms']:.3f} ms",
            flush=True,
        )

    validate_rows(completed_rows)
    selected_keys = {
        (row["experiment"], row["candidate_id"])
        for row in candidates
    }
    completed_selected = sum(
        (int(row["experiment"]), row["candidate_id"]) in selected_keys
        and onnx_result_exists(row)
        for row in completed_rows
    )
    print(
        f"wrote {len(completed_rows)} rows to {args.output}; "
        f"selected experiments completed={completed_selected}/{len(candidates)}"
    )


if __name__ == "__main__":
    main()
