import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import PIIDataset, make_collate_fn
from evaluate import compute_metrics
from nas_encoding import decode_individual, individual_key, normalize_individual
from nas_train import build_nas_model
from train import labels_to_device
from utils import write_json


FIELDNAMES = [
    "experiment",
    "candidate_id",
    "individual",
    "architecture",
    "dev_f1",
    "flops",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_loss",
    "checkpoint",
]


def parse_gpu_list(value: str) -> tuple[str, ...]:
    gpus = tuple(item.strip() for item in value.split(",") if item.strip())
    if not gpus:
        raise ValueError("GPU list must not be empty")
    if len(set(gpus)) != len(gpus):
        raise ValueError("GPU list must not contain duplicates")
    return gpus


def collect_candidates(nas_root: Path, experiments: list[int]) -> list[dict]:
    candidates = []
    for experiment in experiments:
        experiment_dir = nas_root / f"experiment_{experiment}"
        state_path = experiment_dir / "search_state.json"
        if not state_path.exists():
            raise FileNotFoundError(f"search state not found: {state_path}")
        state = json.loads(state_path.read_text(encoding="utf-8"))
        records = state["all_results"]
        records = list(records.values()) if isinstance(records, dict) else records
        for record in records:
            individual = normalize_individual(record["individual"], experiment)
            candidate_id = individual_key(individual, experiment)
            checkpoint = experiment_dir / "candidates" / candidate_id / "best.pt"
            if not checkpoint.is_file():
                raise FileNotFoundError(f"candidate checkpoint not found: {checkpoint}")
            candidates.append(
                {
                    "experiment": experiment,
                    "candidate_id": candidate_id,
                    "individual": list(individual),
                    "architecture": decode_individual(individual, experiment),
                    "dev_f1": record["dev_f1"],
                    "flops": record["flops"],
                    "checkpoint": checkpoint,
                }
            )
    candidates.sort(key=lambda row: (row["experiment"], -row["flops"], row["candidate_id"]))
    keys = [(row["experiment"], row["candidate_id"]) for row in candidates]
    if len(keys) != len(set(keys)):
        raise ValueError("search state contains duplicate normalized candidates")
    return candidates


def evaluate_metrics(model, dataloader, device, id2entity) -> dict:
    model.eval()
    gold_entities = []
    predicted_entities = []
    total_loss = 0.0
    total_steps = 0
    with torch.inference_mode():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = labels_to_device(batch, "cascade", device)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=device.type == "cuda",
            ):
                output = model(input_ids, labels, mask)
            total_loss += output["loss"].item()
            total_steps += 1
            predictions = model.head.decode(
                output["logits"],
                mask,
                id2entity,
                batch["texts"],
            )
            gold_entities.extend(batch["entities"])
            predicted_entities.extend(predictions)
    metrics = compute_metrics(gold_entities, predicted_entities)
    metrics["loss"] = total_loss / max(total_steps, 1)
    return metrics


def evaluate_candidate(candidate: dict, result_path: Path, batch_size: int, cpu: bool) -> dict:
    checkpoint = torch.load(
        candidate["checkpoint"],
        map_location="cpu",
        weights_only=False,
    )
    experiment = checkpoint["experiment"]
    individual = normalize_individual(checkpoint["individual"], experiment)
    candidate_id = individual_key(individual, experiment)
    if experiment != candidate["experiment"] or candidate_id != candidate["candidate_id"]:
        raise ValueError(
            f"checkpoint identity mismatch: expected experiment={candidate['experiment']} "
            f"candidate={candidate['candidate_id']}, got experiment={experiment} "
            f"candidate={candidate_id}"
        )

    cfg = checkpoint["config"]
    entity2id = checkpoint["entity2id"]
    id2entity = {index: name for name, index in entity2id.items()}
    dataset = PIIDataset(
        ROOT / cfg["data"]["test_path"],
        checkpoint["char2id"],
        checkpoint["label2id"],
        cfg["train"]["max_len"],
    )
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(
            entity2id,
            cfg["model"]["cascade_max_span_len"],
        ),
    )
    model = build_nas_model(
        cfg,
        vocab_size=len(checkpoint["char2id"]),
        entity_type_num=len(entity2id),
        individual=individual,
        experiment=experiment,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    metrics = evaluate_metrics(model, dataloader, device, id2entity)
    micro = metrics["micro"]
    result = {
        "experiment": experiment,
        "candidate_id": candidate_id,
        "individual": list(individual),
        "architecture": decode_individual(individual, experiment),
        "dev_f1": candidate["dev_f1"],
        "flops": candidate["flops"],
        "test_precision": micro["precision"],
        "test_recall": micro["recall"],
        "test_f1": micro["f1"],
        "test_loss": metrics["loss"],
        "checkpoint": str(candidate["checkpoint"].resolve()),
    }
    write_json(result, result_path)
    return result


def result_path(output_dir: Path, candidate: dict) -> Path:
    return (
        output_dir
        / f"experiment_{candidate['experiment']}"
        / f"{candidate['candidate_id']}.json"
    )


def read_completed(candidates: list[dict], output_dir: Path) -> tuple[list[dict], list[dict]]:
    completed = []
    pending = []
    for candidate in candidates:
        path = result_path(output_dir, candidate)
        if path.exists():
            row = json.loads(path.read_text(encoding="utf-8"))
            if (
                row["experiment"] != candidate["experiment"]
                or row["candidate_id"] != candidate["candidate_id"]
            ):
                raise ValueError(f"result identity mismatch: {path}")
            completed.append(row)
        else:
            pending.append(candidate)
    return completed, pending


def write_csv(path: Path, rows: list[dict]) -> None:
    keys = [(row["experiment"], row["candidate_id"]) for row in rows]
    if len(keys) != len(set(keys)):
        raise ValueError("Test result rows contain duplicate candidates")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=FIELDNAMES)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: (item["experiment"], item["candidate_id"])):
            writer.writerow(
                {
                    **row,
                    "individual": json.dumps(row["individual"], ensure_ascii=False),
                    "architecture": json.dumps(
                        row["architecture"],
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                }
            )
    temporary.replace(path)


def worker_command(candidate: dict, path: Path, batch_size: int, cpu: bool) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-candidate",
        json.dumps(
            {
                **candidate,
                "checkpoint": str(candidate["checkpoint"]),
            },
            ensure_ascii=False,
        ),
        "--worker-result",
        str(path),
        "--batch-size",
        str(batch_size),
    ]
    if cpu:
        command.append("--cpu")
    return command


def run_parallel(
    candidates: list[dict],
    output_dir: Path,
    batch_size: int,
    gpus: tuple[str, ...] | None,
    cpu: bool,
) -> None:
    pending = list(candidates)
    active = []
    available = list(gpus or ("cpu",))
    completed = 0
    total = len(candidates)
    started = time.time()
    while pending or active:
        while pending and available:
            device_id = available.pop(0)
            candidate = pending.pop(0)
            path = result_path(output_dir, candidate)
            log_path = path.with_suffix(".log")
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_file = log_path.open("w", encoding="utf-8")
            environment = os.environ.copy()
            if device_id != "cpu":
                environment["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
                environment["CUDA_VISIBLE_DEVICES"] = device_id
            process = subprocess.Popen(
                worker_command(candidate, path, batch_size, cpu),
                cwd=ROOT,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
            )
            active.append((process, log_file, candidate, device_id, log_path))

        for item in list(active):
            process, log_file, candidate, device_id, log_path = item
            return_code = process.poll()
            if return_code is None:
                continue
            log_file.close()
            active.remove(item)
            available.append(device_id)
            if return_code != 0:
                raise RuntimeError(
                    f"Test evaluation failed for {candidate['candidate_id']}; "
                    f"see {log_path}"
                )
            completed += 1
            elapsed = time.time() - started
            eta = elapsed / completed * (total - completed)
            row = json.loads(result_path(output_dir, candidate).read_text(encoding="utf-8"))
            print(
                f"[{completed}/{total}] experiment={candidate['experiment']} "
                f"candidate={candidate['candidate_id']} test_f1={row['test_f1']:.6f} "
                f"elapsed={elapsed / 60:.1f}min eta={eta / 60:.1f}min",
                flush=True,
            )
        if active:
            time.sleep(0.2)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--gpus", type=parse_gpu_list, default=("0",))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=ROOT / "outputs" / "nas_test_all",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "outputs" / "reports" / "nas_all_candidates_test.csv",
    )
    parser.add_argument("--worker-candidate")
    parser.add_argument("--worker-result", type=Path)
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.cpu:
        args.gpus = None
    return args


def main() -> None:
    args = parse_args()
    if args.worker_candidate:
        if args.worker_result is None:
            raise ValueError("--worker-result is required in worker mode")
        candidate = json.loads(args.worker_candidate)
        candidate["checkpoint"] = Path(candidate["checkpoint"])
        result = evaluate_candidate(
            candidate,
            args.worker_result,
            args.batch_size,
            args.cpu,
        )
        print(json.dumps(result, ensure_ascii=False))
        return

    experiments = [1, 2, 3] if args.experiment == "all" else [int(args.experiment)]
    candidates = collect_candidates(ROOT / "outputs" / "nas", experiments)
    completed, pending = read_completed(candidates, args.results_dir)
    if args.limit is not None:
        pending = pending[: args.limit]
    print(
        f"unique candidates={len(candidates)} completed={len(completed)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if pending:
        run_parallel(
            pending,
            args.results_dir,
            args.batch_size,
            args.gpus,
            args.cpu,
        )
    completed, still_pending = read_completed(candidates, args.results_dir)
    write_csv(args.output, completed)
    print(
        f"wrote {len(completed)} rows to {args.output}; "
        f"remaining={len(still_pending)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
