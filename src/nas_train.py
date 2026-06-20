import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PIIDataset, load_vocabs, make_collate_fn
from heads import CascadePointerHead
from model_nas_idcnn import NASIDCNNEncoder, NASIDCNNForNER
from nas_archive import transfer_partial_weights
from nas_encoding import decode_individual, individual_key, normalize_individual
from train import apply_token_dropout, entity_types_from_label2id, evaluate
from utils import UNK_TOKEN, ensure_dirs, load_yaml, make_logger, set_seed, write_json


ROOT = Path(__file__).resolve().parents[1]


def candidate_data_paths(cfg: dict, root: Path) -> dict[str, Path]:
    return {
        "processed_dir": root / cfg["data"]["processed_dir"],
        "train": root / cfg["data"]["train_path"],
        "dev": root / cfg["data"]["dev_path"],
    }


def build_nas_model(
    cfg: dict,
    vocab_size: int,
    entity_type_num: int,
    individual,
    experiment: int,
) -> NASIDCNNForNER:
    architecture = decode_individual(individual, experiment)
    encoder = NASIDCNNEncoder(
        vocab_size=vocab_size,
        channels=architecture["C"],
        ratio=architecture["ratio"],
        cell_num=architecture["cell_num"],
        operations=architecture["ops"],
        input_dropout=cfg["model"]["input_dropout"],
        hidden_dropout=cfg["model"]["hidden_dropout"],
    )
    head = CascadePointerHead(
        architecture["C"],
        entity_type_num,
        pointer_size=cfg["model"]["cascade_pointer_size"],
        max_span_len=cfg["model"]["cascade_max_span_len"],
    )
    return NASIDCNNForNER(encoder, head)


def count_nas_flops(model: nn.Module, sequence_length: int = 128) -> int:
    device = next(model.parameters()).device
    input_ids = torch.ones(1, sequence_length, dtype=torch.long, device=device)
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    macs = 0
    handles = []

    def count(module: nn.Module, _inputs: tuple, output: torch.Tensor) -> None:
        nonlocal macs
        if isinstance(module, nn.Conv1d):
            macs += output.numel() * module.kernel_size[0] * module.in_channels // module.groups
        elif isinstance(module, nn.Linear):
            macs += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            model(input_ids, mask=mask)
    finally:
        for handle in handles:
            handle.remove()
    max_span_len = min(sequence_length, model.head.max_span_len)
    span_pairs = sum(sequence_length - offset for offset in range(max_span_len))
    macs += span_pairs * model.head.start_query.out_features
    return 2 * macs


def apply_archive_initialization(
    target_model,
    source_checkpoint: Path,
    target_individual,
    source_individual,
    cfg: dict,
    vocab_size: int,
    entity_type_num: int,
    experiment: int,
) -> str | None:
    checkpoint = torch.load(source_checkpoint, map_location="cpu", weights_only=False)
    source_model = build_nas_model(
        cfg,
        vocab_size,
        entity_type_num,
        source_individual,
        experiment,
    )
    source_model.load_state_dict(checkpoint["model_state_dict"])
    return transfer_partial_weights(
        target_model,
        source_model,
        target_individual,
        source_individual,
        experiment,
    )


def train_candidate(args) -> dict:
    started = time.time()
    cfg = load_yaml(ROOT / args.config)
    set_seed(42)
    individual = normalize_individual(args.individual, args.experiment)
    output_dir = Path(args.output_dir)
    ensure_dirs(output_dir)
    logger = make_logger(output_dir / "train.log")
    paths = candidate_data_paths(cfg, ROOT)
    char2id, label2id = load_vocabs(paths["processed_dir"])
    entity_types = entity_types_from_label2id(label2id)
    entity2id = {name: index for index, name in enumerate(entity_types)}
    id2label = {index: label for label, index in label2id.items()}
    id2entity = {index: name for name, index in entity2id.items()}

    train_ds = PIIDataset(paths["train"], char2id, label2id, args.max_len)
    dev_ds = PIIDataset(paths["dev"], char2id, label2id, args.max_len)
    collate_fn = make_collate_fn(entity2id, cfg["model"]["cascade_max_span_len"])
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
    )
    dev_loader = DataLoader(
        dev_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )

    model = build_nas_model(cfg, len(char2id), len(entity2id), individual, args.experiment)
    transferred_module = None
    if args.source_checkpoint:
        transferred_module = apply_archive_initialization(
            model,
            Path(args.source_checkpoint),
            individual,
            args.source_individual,
            cfg,
            len(char2id),
            len(entity2id),
            args.experiment,
        )
    flops = count_nas_flops(model, sequence_length=128)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    model.to(device)
    optimizer = Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.epsilon,
        weight_decay=args.weight_decay,
    )
    unk_id = char2id[UNK_TOKEN]
    best_f1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    history = []
    checkpoint_path = output_dir / "best.pt"

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"{output_dir.name} epoch {epoch}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            input_ids = apply_token_dropout(input_ids, mask, unk_id, args.token_dropout)
            labels = {name: value.to(device) for name, value in batch["cascade_labels"].items()}
            loss = model(input_ids, labels, mask)["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            steps += 1

        dev_metrics, _ = evaluate(
            model,
            dev_loader,
            device,
            id2label,
            id2entity,
            head="cascade",
        )
        dev_micro = dev_metrics["micro"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(steps, 1),
                "dev_precision": dev_micro["precision"],
                "dev_recall": dev_micro["recall"],
                "dev_f1": dev_micro["f1"],
            }
        )
        logger.info("epoch=%d dev_f1=%.6f", epoch, dev_micro["f1"])
        if dev_micro["f1"] > best_f1:
            best_f1 = dev_micro["f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "individual": list(individual),
                    "experiment": args.experiment,
                    "char2id": char2id,
                    "label2id": label2id,
                    "entity2id": entity2id,
                    "best_epoch": best_epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_metrics, _ = evaluate(model, dev_loader, device, id2label, id2entity, head="cascade")
    result = {
        "candidate_id": individual_key(individual, args.experiment),
        "individual": list(individual),
        "architecture": decode_individual(individual, args.experiment),
        "experiment": args.experiment,
        "seed": 42,
        "best_epoch": best_epoch,
        "dev": dev_metrics,
        "dev_f1": dev_metrics["micro"]["f1"],
        "flops": flops,
        "checkpoint": str(checkpoint_path.resolve()),
        "source_checkpoint": args.source_checkpoint,
        "transferred_module": transferred_module,
        "history": history,
        "train_seconds": time.time() - started,
    }
    write_json(result, output_dir / "candidate.json")
    print(json.dumps(result, ensure_ascii=False))
    return result


def parse_individual(value: str) -> tuple[int, ...]:
    genes = tuple(int(item) for item in value.split(","))
    if len(genes) != 12:
        raise argparse.ArgumentTypeError("individual must contain 12 comma-separated integers")
    return genes


def parse_args():
    cfg = load_yaml(ROOT / "configs.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--experiment", type=int, required=True, choices=[1, 2, 3])
    parser.add_argument("--individual", type=parse_individual, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-checkpoint")
    parser.add_argument("--source-individual", type=parse_individual)
    parser.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
    parser.add_argument("--batch-size", type=int, default=cfg["train"]["batch_size"])
    parser.add_argument("--max-len", type=int, default=cfg["train"]["max_len"])
    parser.add_argument("--lr", type=float, default=cfg["train"]["lr"])
    parser.add_argument("--beta1", type=float, default=cfg["train"]["beta1"])
    parser.add_argument("--beta2", type=float, default=cfg["train"]["beta2"])
    parser.add_argument("--epsilon", type=float, default=cfg["train"]["epsilon"])
    parser.add_argument("--weight-decay", type=float, default=cfg["train"]["weight_decay"])
    parser.add_argument("--grad-clip", type=float, default=cfg["train"]["grad_clip"])
    parser.add_argument("--token-dropout", type=float, default=cfg["train"]["token_dropout"])
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if bool(args.source_checkpoint) != bool(args.source_individual):
        parser.error("--source-checkpoint and --source-individual must be provided together")
    return args


if __name__ == "__main__":
    train_candidate(parse_args())
