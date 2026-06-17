import argparse
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils import PAD_TOKEN, UNK_TOKEN, load_yaml, read_json, write_json, write_jsonl


def validate_sample(sample: dict) -> tuple[list[str], list[dict]]:
    text = sample.get("text", "")
    errors = []
    normalized = []
    spans = []

    for ent in sample.get("entities", []):
        start = ent.get("start")
        end = ent.get("end")
        ent_text = ent.get("text", "")
        ent_type = ent.get("type")
        if not isinstance(start, int) or not isinstance(end, int) or start < 0 or end > len(text) or start >= end:
            errors.append({"error": "span_out_of_bounds", "entity": ent})
            continue
        if text[start:end] != ent_text:
            errors.append({"error": "text_mismatch", "entity": ent, "actual": text[start:end]})
            continue
        normalized.append({"text": ent_text, "type": ent_type, "start": start, "end": end})
        spans.append((start, end, ent))

    spans.sort(key=lambda x: (x[0], x[1]))
    for (_, prev_end, prev_ent), (start, _, ent) in zip(spans, spans[1:]):
        if start < prev_end:
            errors.append({"error": "span_overlap", "entity": ent, "previous": prev_ent})

    if errors:
        return errors, []
    normalized.sort(key=lambda x: (x["start"], x["end"]))
    return [], normalized


def to_bilou(text: str, entities: list[dict]) -> list[str]:
    labels = ["O"] * len(text)
    for ent in entities:
        start, end, ent_type = ent["start"], ent["end"], ent["type"]
        length = end - start
        if length == 1:
            labels[start] = f"U-{ent_type}"
        else:
            labels[start] = f"B-{ent_type}"
            for i in range(start + 1, end - 1):
                labels[i] = f"I-{ent_type}"
            labels[end - 1] = f"L-{ent_type}"
    return labels


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs.yaml")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    raw_path = ROOT / cfg["data"]["raw_path"]
    processed_dir = ROOT / cfg["data"]["processed_dir"]
    log_path = ROOT / "logs" / "invalid_spans.jsonl"
    processed_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    raw = read_json(raw_path)
    valid_rows = []
    invalid_rows = []
    entity_types = set()
    chars = set()

    for idx, sample in enumerate(raw):
        errors, entities = validate_sample(sample)
        if errors:
            invalid_rows.append({"index": idx, "id": sample.get("id"), "errors": errors, "sample": sample})
            continue
        text = sample["text"]
        labels = to_bilou(text, entities)
        valid_rows.append({"id": sample.get("id", idx), "text": text, "entities": entities, "labels": labels})
        chars.update(text)
        entity_types.update(ent["type"] for ent in entities)

    write_jsonl(invalid_rows, log_path)

    rng = random.Random(args.seed)
    rng.shuffle(valid_rows)
    n = len(valid_rows)
    n_train = int(n * 0.8)
    n_dev = int(n * 0.1)
    splits = {
        "train": valid_rows[:n_train],
        "dev": valid_rows[n_train : n_train + n_dev],
        "test": valid_rows[n_train + n_dev :],
    }
    for name, rows in splits.items():
        write_jsonl(rows, processed_dir / f"{name}.jsonl")

    char2id = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for ch in sorted(chars):
        char2id[ch] = len(char2id)

    labels = ["O"]
    for ent_type in sorted(entity_types):
        labels.extend([f"B-{ent_type}", f"I-{ent_type}", f"L-{ent_type}", f"U-{ent_type}"])
    label2id = {label: i for i, label in enumerate(labels)}

    write_json(char2id, processed_dir / "char2id.json")
    write_json(label2id, processed_dir / "label2id.json")
    print(
        f"valid={len(valid_rows)} invalid={len(invalid_rows)} "
        f"train={len(splits['train'])} dev={len(splits['dev'])} test={len(splits['test'])}"
    )


if __name__ == "__main__":
    main()
