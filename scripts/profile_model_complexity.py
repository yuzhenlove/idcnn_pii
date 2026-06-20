import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from evaluate import labels_to_entities
from heads import CascadePointerHead, EfficientGlobalPointerHead
from train import build_model


TEST_TEXT = "经理您好，我是冯璐（工号EMP-5813-CV），因感冒发烧无法到岗，申请病假一天，已邮件请假。手机18659482211畅通。"


def checkpoint_display_path(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT))


def count_forward_macs(model: nn.Module, args: tuple, kwargs: dict | None = None) -> int:
    total = 0
    handles = []

    def count(module: nn.Module, inputs: tuple, output: torch.Tensor) -> None:
        nonlocal total
        if isinstance(module, nn.Conv1d):
            output_values = output.numel()
            total += output_values * module.kernel_size[0] * module.in_channels // module.groups
        elif isinstance(module, nn.Linear):
            total += output.numel() * module.in_features

    for module in model.modules():
        if isinstance(module, (nn.Conv1d, nn.Linear)):
            handles.append(module.register_forward_hook(count))
    try:
        with torch.inference_mode():
            model(*args, **(kwargs or {}))
    finally:
        for handle in handles:
            handle.remove()
    return total


def extra_head_macs(model: nn.Module, seq_len: int, batch_size: int = 1) -> int:
    head = model.head
    if isinstance(head, EfficientGlobalPointerHead):
        return batch_size * seq_len * seq_len * head.head_size
    if isinstance(head, CascadePointerHead):
        span_pairs = sum(seq_len - offset for offset in range(min(seq_len, head.max_span_len)))
        return batch_size * span_pairs * head.start_query.out_features
    return 0


def decode_entities(model, output, mask, checkpoint):
    head_name = checkpoint["args"]["head"]
    if head_name in {"egp", "cascade"}:
        id2entity = {idx: name for name, idx in checkpoint["entity2id"].items()}
        return model.head.decode(output["logits"], mask, id2entity, [TEST_TEXT])
    id2label = {idx: name for name, idx in checkpoint["label2id"].items()}
    labels = model.head.decode(output["logits"], mask, id2label)
    return [labels_to_entities(row, TEST_TEXT) for row in labels]


def measure_complete_inference(model, input_ids, mask, checkpoint, warmup: int, iterations: int):
    device = input_ids.device

    def run_once():
        with torch.inference_mode():
            output = model(input_ids, mask=mask)
            decode_entities(model, output, mask, checkpoint)

    for _ in range(warmup):
        run_once()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        run_once()
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        timings.append((time.perf_counter() - start) * 1000)

    mean_ms = float(np.mean(timings))
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2 if device.type == "cuda" else 0.0
    return mean_ms, float(np.percentile(timings, 50)), float(np.percentile(timings, 95)), peak_mb


def profile_checkpoint(path: Path, device: torch.device, warmup: int, iterations: int) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    head_name = args["head"]
    num_blocks = args["num_blocks"]
    output_size = len(checkpoint["entity2id"]) if head_name in {"egp", "cascade"} else len(checkpoint["label2id"])
    model = build_model(
        checkpoint["config"],
        len(checkpoint["char2id"]),
        output_size,
        num_blocks,
        head_name,
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    unk_id = checkpoint["char2id"]["<UNK>"]
    input_ids = torch.tensor(
        [[checkpoint["char2id"].get(char, unk_id) for char in TEST_TEXT]],
        dtype=torch.long,
        device=device,
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    macs = count_forward_macs(model, (input_ids,), {"mask": mask})
    macs += extra_head_macs(model, len(TEST_TEXT))
    mean_ms, p50_ms, p95_ms, peak_mb = measure_complete_inference(
        model, input_ids, mask, checkpoint, warmup, iterations
    )
    row = {
        "run_id": path.parent.name,
        "checkpoint": checkpoint_display_path(path),
        "head": head_name,
        "num_blocks": num_blocks,
        "seed": args["seed"],
        "input_length": len(TEST_TEXT),
        "batch_size": 1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "macs": macs,
        "flops": 2 * macs,
        "mean_latency_ms": mean_ms,
        "p50_latency_ms": p50_ms,
        "p95_latency_ms": p95_ms,
        "throughput_samples_per_second": 1000.0 / mean_ms,
        "peak_gpu_memory_mb": peak_mb,
        "device": str(device),
    }
    del model, checkpoint, input_ids, mask
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    checkpoints = sorted((ROOT / "outputs").glob("*_b*_seed*/best.pt"))
    if len(checkpoints) != 48:
        raise RuntimeError(f"expected 48 checkpoints, found {len(checkpoints)}")

    rows = []
    for index, path in enumerate(checkpoints, 1):
        row = profile_checkpoint(path, device, args.warmup, args.iterations)
        rows.append(row)
        print(f"[{index:02d}/48] {row['run_id']} mean={row['mean_latency_ms']:.3f} ms")

    output_path = ROOT / "outputs" / "reports" / "model_complexity_64.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
