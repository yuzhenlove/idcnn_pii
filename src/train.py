import argparse
import json
from pathlib import Path

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import PIIDataset, make_collate_fn, load_vocabs
from evaluate import compute_metrics, labels_to_entities
from heads import CRFHead, EfficientGlobalPointerHead, SoftmaxHead
from model_idcnn import IDCNNEncoder, IDCNNForTokenClassification
from utils import UNK_TOKEN, ensure_dirs, load_yaml, make_logger, set_seed, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[1]


def build_model(cfg: dict, vocab_size: int, output_size: int, num_blocks: int, head: str) -> IDCNNForTokenClassification:
    encoder = IDCNNEncoder(
        vocab_size=vocab_size,
        embedding_dim=cfg["model"]["embedding_dim"],
        hidden_size=cfg["model"]["hidden_size"],
        input_dropout=cfg["model"]["input_dropout"],
        hidden_dropout=cfg["model"]["hidden_dropout"],
        dilations=cfg["model"]["dilations"],
        kernel_size=cfg["model"]["kernel_size"],
        num_blocks=num_blocks,
    )
    if head == "softmax":
        token_head = SoftmaxHead(cfg["model"]["hidden_size"], output_size)
    elif head == "crf":
        token_head = CRFHead(cfg["model"]["hidden_size"], output_size)
    elif head == "egp":
        token_head = EfficientGlobalPointerHead(cfg["model"]["hidden_size"], output_size)
    else:
        raise ValueError(f"unsupported head: {head}")
    drop_penalty = 0.0 if head == "egp" else cfg["train"]["drop_penalty"]
    return IDCNNForTokenClassification(encoder, token_head, drop_penalty=drop_penalty)


def apply_token_dropout(
    input_ids: torch.Tensor,
    mask: torch.Tensor,
    unk_id: int,
    probability: float,
) -> torch.Tensor:
    if probability <= 0:
        return input_ids
    drop_mask = (torch.rand(input_ids.shape, device=input_ids.device) < probability) & mask.bool()
    return input_ids.masked_fill(drop_mask, unk_id)


def entity_types_from_label2id(label2id: dict[str, int]) -> list[str]:
    entity_types = set()
    for label in label2id:
        if "-" in label:
            _, ent_type = label.split("-", 1)
            entity_types.add(ent_type)
    return sorted(entity_types)


def evaluate(model, dataloader, device, id2label, id2entity=None, head: str = "softmax"):
    model.eval()
    gold_batches, pred_batches = [], []
    predictions = []
    total_loss = 0.0
    total_steps = 0
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["span_labels"].to(device) if head == "egp" else batch["labels"].to(device)
            out = model(input_ids, labels, mask)
            total_loss += out["loss"].item()
            total_steps += 1
            if head == "egp":
                pred_entities = model.head.decode(out["logits"], mask, id2entity, batch["texts"])
            else:
                decoded = model.head.decode(out["logits"], mask, id2label)
                pred_entities = [labels_to_entities(seq, text) for seq, text in zip(decoded, batch["texts"])]
            gold_batches.extend(batch["entities"])
            pred_batches.extend(pred_entities)
            for row_id, text, gold, pred in zip(batch["ids"], batch["texts"], batch["entities"], pred_entities):
                predictions.append({"id": row_id, "text": text, "gold_entities": gold, "pred_entities": pred})
    metrics = compute_metrics(gold_batches, pred_batches)
    metrics["loss"] = total_loss / max(total_steps, 1)
    return metrics, predictions


def train(args):
    cfg = load_yaml(ROOT / args.config)
    set_seed(args.seed)
    if args.num_blocks not in {1, 2, 3, 4}:
        raise ValueError("--num_blocks must be one of 1, 2, 3, 4")

    run_id = f"{args.head}_b{args.num_blocks}_seed{args.seed}"
    output_dir = ROOT / "outputs" / run_id
    log_path = ROOT / "logs" / run_id / "train.log"
    ensure_dirs(output_dir, log_path.parent)
    logger = make_logger(log_path)

    char2id, label2id = load_vocabs(ROOT / cfg["data"]["processed_dir"])
    id2label = {idx: label for label, idx in label2id.items()}
    entity_types = entity_types_from_label2id(label2id)
    entity2id = {ent_type: idx for idx, ent_type in enumerate(entity_types)}
    id2entity = {idx: ent_type for ent_type, idx in entity2id.items()}
    train_ds = PIIDataset(ROOT / cfg["data"]["train_path"], char2id, label2id, args.max_len)
    dev_ds = PIIDataset(ROOT / cfg["data"]["dev_path"], char2id, label2id, args.max_len)
    test_ds = PIIDataset(ROOT / cfg["data"]["test_path"], char2id, label2id, args.max_len)

    collate_fn = make_collate_fn(entity2id if args.head == "egp" else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_loader = DataLoader(dev_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    output_size = len(entity2id) if args.head == "egp" else len(label2id)
    model = build_model(cfg, len(char2id), output_size, args.num_blocks, args.head).to(device)
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
    logger.info("run_id=%s device=%s train=%d dev=%d test=%d", run_id, device, len(train_ds), len(dev_ds), len(test_ds))

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        steps = 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}", leave=False)
        for batch in progress:
            optimizer.zero_grad()
            input_ids = batch["input_ids"].to(device)
            mask = batch["mask"].to(device)
            input_ids = apply_token_dropout(input_ids, mask, unk_id, args.token_dropout)
            labels = batch["span_labels"].to(device) if args.head == "egp" else batch["labels"].to(device)
            out = model(input_ids, labels, mask)
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            total_loss += loss.item()
            steps += 1
            progress.set_postfix(loss=f"{total_loss / steps:.4f}")

        train_loss = total_loss / max(steps, 1)
        dev_metrics, _ = evaluate(model, dev_loader, device, id2label, id2entity, args.head)
        dev_micro = dev_metrics["micro"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "dev_precision": dev_micro["precision"],
            "dev_recall": dev_micro["recall"],
            "dev_f1": dev_micro["f1"],
        }
        history.append(row)
        logger.info(
            "epoch=%d train_loss=%.6f dev_precision=%.6f dev_recall=%.6f dev_f1=%.6f",
            epoch,
            row["train_loss"],
            row["dev_precision"],
            row["dev_recall"],
            row["dev_f1"],
        )

        if dev_micro["f1"] > best_f1:
            best_f1 = dev_micro["f1"]
            best_epoch = epoch
            stale_epochs = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": cfg,
                    "args": vars(args),
                    "char2id": char2id,
                    "label2id": label2id,
                    "entity2id": entity2id,
                    "best_epoch": best_epoch,
                },
                output_dir / "best.pt",
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.early_stop_patience:
                logger.info("early stop at epoch=%d best_epoch=%d best_f1=%.6f", epoch, best_epoch, best_f1)
                break

    checkpoint = torch.load(output_dir / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    dev_metrics, _ = evaluate(model, dev_loader, device, id2label, id2entity, args.head)
    test_metrics, test_predictions = evaluate(model, test_loader, device, id2label, id2entity, args.head)

    metrics = {
        "run_id": run_id,
        "head": args.head,
        "num_blocks": args.num_blocks,
        "seed": args.seed,
        "best_epoch": best_epoch,
        "history": history,
        "dev": dev_metrics,
        "test": test_metrics,
    }
    write_json(metrics, output_dir / "metrics.json")
    write_jsonl(test_predictions, output_dir / "test_predictions.jsonl")
    logger.info("saved best=%s metrics=%s predictions=%s", output_dir / "best.pt", output_dir / "metrics.json", output_dir / "test_predictions.jsonl")
    logger.info("test_precision=%.6f test_recall=%.6f test_f1=%.6f", test_metrics["micro"]["precision"], test_metrics["micro"]["recall"], test_metrics["micro"]["f1"])
    print(json.dumps({"run_id": run_id, "best_epoch": best_epoch, "test": test_metrics["micro"]}, ensure_ascii=False))


def parse_args():
    cfg = load_yaml(ROOT / "configs.yaml")
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--head", default="softmax", choices=["softmax", "crf", "egp"])
    parser.add_argument("--num_blocks", type=int, default=1, choices=[1, 2, 3, 4])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=cfg["train"]["epochs"])
    parser.add_argument("--batch_size", type=int, default=cfg["train"]["batch_size"])
    parser.add_argument("--max_len", type=int, default=cfg["train"]["max_len"])
    parser.add_argument("--lr", type=float, default=cfg["train"]["lr"])
    parser.add_argument("--beta1", type=float, default=cfg["train"]["beta1"])
    parser.add_argument("--beta2", type=float, default=cfg["train"]["beta2"])
    parser.add_argument("--epsilon", type=float, default=cfg["train"]["epsilon"])
    parser.add_argument("--weight_decay", type=float, default=cfg["train"]["weight_decay"])
    parser.add_argument("--grad_clip", type=float, default=cfg["train"]["grad_clip"])
    parser.add_argument("--token_dropout", type=float, default=cfg["train"]["token_dropout"])
    parser.add_argument("--early_stop_patience", type=int, default=cfg["train"]["early_stop_patience"])
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
