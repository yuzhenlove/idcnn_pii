import argparse
import csv
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from data import PIIDataset, make_collate_fn
from nas_train import build_nas_model
from train import evaluate
from utils import load_yaml, read_json, write_json, write_jsonl


def pareto_test_path(cfg: dict, root: Path) -> Path:
    return root / cfg["data"]["test_path"]


def evaluate_checkpoint(checkpoint_path: Path, batch_size: int, cpu: bool) -> tuple[dict, list[dict]]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = checkpoint["config"]
    char2id = checkpoint["char2id"]
    label2id = checkpoint["label2id"]
    entity2id = checkpoint["entity2id"]
    id2label = {index: label for label, index in label2id.items()}
    id2entity = {index: name for name, index in entity2id.items()}
    test_ds = PIIDataset(
        pareto_test_path(cfg, ROOT),
        char2id,
        label2id,
        cfg["train"]["max_len"],
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=make_collate_fn(entity2id, cfg["model"]["cascade_max_span_len"]),
    )
    model = build_nas_model(
        cfg,
        len(char2id),
        len(entity2id),
        checkpoint["individual"],
        checkpoint["experiment"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cpu" if cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    return evaluate(
        model,
        test_loader,
        device,
        id2label,
        id2entity,
        head="cascade",
        autocast_dtype=torch.bfloat16,
    )


def evaluate_experiment(experiment: int, batch_size: int, cpu: bool) -> list[dict]:
    experiment_dir = ROOT / "outputs" / "nas" / f"experiment_{experiment}"
    pareto_path = experiment_dir / "pareto.json"
    if not pareto_path.exists():
        raise FileNotFoundError(f"search Pareto result not found: {pareto_path}")
    rows = []
    for pareto_row in read_json(pareto_path):
        candidate_id = pareto_row["candidate_id"]
        metrics, predictions = evaluate_checkpoint(
            Path(pareto_row["checkpoint"]),
            batch_size,
            cpu,
        )
        output_dir = experiment_dir / "test" / candidate_id
        write_json(metrics, output_dir / "test_metrics.json")
        write_jsonl(predictions, output_dir / "test_predictions.jsonl")
        rows.append(
            {
                "candidate_id": candidate_id,
                "individual": pareto_row["individual"],
                "dev_f1": pareto_row["dev_f1"],
                "flops": pareto_row["flops"],
                "test_precision": metrics["micro"]["precision"],
                "test_recall": metrics["micro"]["recall"],
                "test_f1": metrics["micro"]["f1"],
                "checkpoint": pareto_row["checkpoint"],
            }
        )
        print(
            f"experiment={experiment} candidate={candidate_id} "
            f"test_f1={metrics['micro']['f1']:.6f}",
            flush=True,
        )
    write_json(rows, experiment_dir / "pareto_test.json")
    with open(experiment_dir / "pareto_test.csv", "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "candidate_id",
                "dev_f1",
                "flops",
                "test_precision",
                "test_recall",
                "test_f1",
                "checkpoint",
                "individual",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> None:
    cfg = load_yaml(ROOT / "configs.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="all", choices=["1", "2", "3", "all"])
    parser.add_argument("--batch-size", type=int, default=cfg["train"]["batch_size"])
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    experiments = [3, 1, 2] if args.experiment == "all" else [int(args.experiment)]
    for experiment in experiments:
        evaluate_experiment(experiment, args.batch_size, args.cpu)


if __name__ == "__main__":
    main()
