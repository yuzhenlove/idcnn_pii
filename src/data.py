from pathlib import Path

import torch
from torch.utils.data import Dataset

from utils import PAD_LABEL_ID, read_json, read_jsonl


class PIIDataset(Dataset):
    def __init__(self, path: str | Path, char2id: dict[str, int], label2id: dict[str, int], max_len: int = 256):
        self.rows = read_jsonl(path)
        self.char2id = char2id
        self.label2id = label2id
        self.max_len = max_len
        self.unk_id = char2id["<UNK>"]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        row = self.rows[idx]
        text = row["text"][: self.max_len]
        labels = row["labels"][: self.max_len]
        input_ids = [self.char2id.get(ch, self.unk_id) for ch in text]
        label_ids = [self.label2id[label] for label in labels]
        return {
            "id": row.get("id", idx),
            "text": text,
            "input_ids": input_ids,
            "label_ids": label_ids,
            "entities": [e for e in row.get("entities", []) if e["end"] <= self.max_len],
        }


def collate_batch(
    batch: list[dict],
    entity2id: dict[str, int] | None = None,
    cascade_max_span_len: int | None = None,
) -> dict:
    max_len = max(len(item["input_ids"]) for item in batch)
    input_ids, labels, masks = [], [], []
    for item in batch:
        n = len(item["input_ids"])
        pad = max_len - n
        input_ids.append(item["input_ids"] + [0] * pad)
        labels.append(item["label_ids"] + [PAD_LABEL_ID] * pad)
        masks.append([1] * n + [0] * pad)
    result = {
        "ids": [item["id"] for item in batch],
        "texts": [item["text"] for item in batch],
        "entities": [item["entities"] for item in batch],
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "mask": torch.tensor(masks, dtype=torch.bool),
    }
    if entity2id is not None and cascade_max_span_len is not None:
        start_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        end_labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
        for batch_idx, item in enumerate(batch):
            start_labels[batch_idx, : len(item["input_ids"])] = 0
            for ent in item["entities"]:
                span_len = ent["end"] - ent["start"]
                if ent["type"] in entity2id and span_len <= cascade_max_span_len:
                    start_labels[batch_idx, ent["start"]] = entity2id[ent["type"]] + 1
                    end_labels[batch_idx, ent["start"]] = span_len - 1
        result["cascade_labels"] = {
            "start_labels": start_labels,
            "end_labels": end_labels,
        }
    elif entity2id is not None:
        span_labels = torch.zeros(len(batch), len(entity2id), max_len, max_len, dtype=torch.float)
        for batch_idx, item in enumerate(batch):
            for ent in item["entities"]:
                if ent["type"] in entity2id and ent["end"] <= max_len:
                    span_labels[batch_idx, entity2id[ent["type"]], ent["start"], ent["end"] - 1] = 1.0
        result["span_labels"] = span_labels
    return result


def make_collate_fn(
    entity2id: dict[str, int] | None = None,
    cascade_max_span_len: int | None = None,
):
    def _collate(batch: list[dict]) -> dict:
        return collate_batch(batch, entity2id, cascade_max_span_len)

    return _collate


def load_vocabs(processed_dir: str | Path) -> tuple[dict[str, int], dict[str, int]]:
    processed_dir = Path(processed_dir)
    return read_json(processed_dir / "char2id.json"), read_json(processed_dir / "label2id.json")
