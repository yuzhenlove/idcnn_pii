from collections import Counter, defaultdict
from typing import Iterable


def labels_to_entities(labels: list[str], text: str | None = None) -> list[dict]:
    entities = []
    active_type = None
    active_start = None

    def close(end: int) -> None:
        nonlocal active_type, active_start
        if active_type is not None and active_start is not None:
            item = {"type": active_type, "start": active_start, "end": end}
            if text is not None:
                item["text"] = text[active_start:end]
            entities.append(item)
        active_type = None
        active_start = None

    for i, label in enumerate(labels):
        if label == "O" or label == "<PAD>":
            close(i)
            continue
        if "-" not in label:
            close(i)
            continue
        prefix, ent_type = label.split("-", 1)
        if prefix == "U":
            close(i)
            item = {"type": ent_type, "start": i, "end": i + 1}
            if text is not None:
                item["text"] = text[i : i + 1]
            entities.append(item)
        elif prefix == "B":
            close(i)
            active_type = ent_type
            active_start = i
        elif prefix == "I":
            if active_type != ent_type:
                close(i)
                active_type = ent_type
                active_start = i
        elif prefix == "L":
            if active_type == ent_type:
                close(i + 1)
            else:
                close(i)
                item = {"type": ent_type, "start": i, "end": i + 1}
                if text is not None:
                    item["text"] = text[i : i + 1]
                entities.append(item)
        else:
            close(i)
    close(len(labels))
    return entities


def entity_set(entities: Iterable[dict]) -> set[tuple[str, int, int]]:
    return {(e["type"], int(e["start"]), int(e["end"])) for e in entities}


def prf(tp: int, pred: int, gold: int) -> dict[str, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / gold if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_metrics(gold_batches: list[list[dict]], pred_batches: list[list[dict]]) -> dict:
    total_tp = total_pred = total_gold = 0
    per_type = defaultdict(lambda: Counter(tp=0, pred=0, gold=0))

    for gold_entities, pred_entities in zip(gold_batches, pred_batches):
        gold = entity_set(gold_entities)
        pred = entity_set(pred_entities)
        hits = gold & pred
        total_tp += len(hits)
        total_pred += len(pred)
        total_gold += len(gold)

        for ent_type, _, _ in gold:
            per_type[ent_type]["gold"] += 1
        for ent_type, _, _ in pred:
            per_type[ent_type]["pred"] += 1
        for ent_type, _, _ in hits:
            per_type[ent_type]["tp"] += 1

    metrics = {"micro": prf(total_tp, total_pred, total_gold), "per_type": {}}
    for ent_type in sorted(per_type):
        c = per_type[ent_type]
        metrics["per_type"][ent_type] = prf(c["tp"], c["pred"], c["gold"])
    return metrics
