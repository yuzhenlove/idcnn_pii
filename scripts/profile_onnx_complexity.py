import argparse
import csv
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from evaluate import labels_to_entities
from train import build_model

from scripts.profile_model_complexity import (
    TEST_TEXT,
    checkpoint_display_path,
    count_forward_macs,
    extra_head_macs,
)


class OnnxExportWrapper(nn.Module):
    def __init__(self, model: nn.Module, head: str):
        super().__init__()
        self.model = model
        self.head = head

    def forward(self, input_ids: torch.Tensor, mask: torch.Tensor):
        if self.head == "egp":
            features = self.model.encoder(input_ids, mask)
            head = self.model.head
            qk = head.qk_proj(features)
            qw, kw = qk[..., : head.head_size], qk[..., head.head_size :]
            qw, kw = self.apply_rope(qw), self.apply_rope(kw)
            logits = torch.matmul(qw, kw.transpose(-1, -2)) / (head.head_size**0.5)
            bias = head.bias_proj(features).view(features.size(0), features.size(1), head.entity_type_num, 2)
            entity_logits = []
            for entity_id in range(head.entity_type_num):
                start_bias = bias[:, :, entity_id, 0].unsqueeze(-1)
                end_bias = bias[:, :, entity_id, 1].unsqueeze(1)
                entity_logits.append((logits + start_bias + end_bias).unsqueeze(1))
            return torch.cat(entity_logits, dim=1)
        logits = self.model(input_ids, mask=mask)["logits"]
        if self.head == "cascade":
            return logits["start"], logits["end"]
        return logits

    @staticmethod
    def apply_rope(x: torch.Tensor) -> torch.Tensor:
        position = torch.arange(x.size(1), dtype=x.dtype, device=x.device).unsqueeze(1)
        indices = torch.arange(0, x.size(2), 2, dtype=x.dtype, device=x.device)
        sinusoid = position * torch.pow(10000.0, -indices / x.size(2))
        sin, cos = sinusoid.sin().unsqueeze(0), sinusoid.cos().unsqueeze(0)
        even, odd = x[..., 0::2], x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = odd * cos + even * sin
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def latency_stats(seconds: list[float]) -> dict[str, float]:
    milliseconds = np.asarray(seconds) * 1000.0
    return {
        "mean_ms": float(milliseconds.mean()),
        "p50_ms": float(np.percentile(milliseconds, 50)),
        "p95_ms": float(np.percentile(milliseconds, 95)),
    }


def measure(fn, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        fn()
    timings = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        timings.append(time.perf_counter() - start)
    return latency_stats(timings)


def export_onnx(model, head: str, input_ids, mask, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_names = ["start_logits", "end_logits"] if head == "cascade" else ["logits"]
    wrapper = OnnxExportWrapper(model, head).eval()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        torch.onnx.export(
            wrapper,
            (input_ids, mask),
            output_path,
            input_names=["input_ids", "mask"],
            output_names=output_names,
            opset_version=17,
            dynamo=False,
        )
    model.eval()


def make_session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(path, sess_options=options, providers=["CPUExecutionProvider"])


def restore_logits(head: str, outputs: list[np.ndarray]):
    tensors = [torch.from_numpy(output) for output in outputs]
    if head == "cascade":
        return {"start": tensors[0], "end": tensors[1]}
    return tensors[0]


def decode_entities(model, head: str, outputs, mask, checkpoint):
    logits = restore_logits(head, outputs)
    if head in {"egp", "cascade"}:
        id2entity = {idx: name for name, idx in checkpoint["entity2id"].items()}
        return model.head.decode(logits, mask, id2entity, [TEST_TEXT])
    id2label = {idx: name for name, idx in checkpoint["label2id"].items()}
    labels = model.head.decode(logits, mask, id2label)
    return [labels_to_entities(row, TEST_TEXT) for row in labels]


def profile_checkpoint(path: Path, onnx_dir: Path, warmup: int, iterations: int) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    args = checkpoint["args"]
    head = args["head"]
    output_size = len(checkpoint["entity2id"]) if head in {"egp", "cascade"} else len(checkpoint["label2id"])
    model = build_model(
        checkpoint["config"],
        len(checkpoint["char2id"]),
        output_size,
        args["num_blocks"],
        head,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    unk_id = checkpoint["char2id"]["<UNK>"]
    input_ids = torch.tensor(
        [[checkpoint["char2id"].get(char, unk_id) for char in TEST_TEXT]],
        dtype=torch.long,
    )
    mask = torch.ones_like(input_ids, dtype=torch.bool)
    onnx_path = onnx_dir / f"{path.parent.name}.onnx"
    export_onnx(model, head, input_ids, mask, onnx_path)
    session = make_session(onnx_path)
    feeds = {"input_ids": input_ids.numpy(), "mask": mask.numpy()}

    with torch.inference_mode():
        torch_output = OnnxExportWrapper(model, head)(input_ids, mask)
    torch_outputs = list(torch_output) if head == "cascade" else [torch_output]
    onnx_outputs = session.run(None, feeds)
    for expected, actual in zip(torch_outputs, onnx_outputs):
        np.testing.assert_allclose(expected.detach().numpy(), actual, rtol=1e-4, atol=1e-4)

    cached_outputs = session.run(None, feeds)
    forward = measure(lambda: session.run(None, feeds), warmup, iterations)
    decoding = measure(
        lambda: decode_entities(model, head, cached_outputs, mask, checkpoint),
        warmup,
        iterations,
    )

    def complete_inference():
        outputs = session.run(None, feeds)
        decode_entities(model, head, outputs, mask, checkpoint)

    complete = measure(complete_inference, warmup, iterations)
    macs = count_forward_macs(model, (input_ids,), {"mask": mask})
    macs += extra_head_macs(model, len(TEST_TEXT))
    return {
        "run_id": path.parent.name,
        "checkpoint": checkpoint_display_path(path),
        "onnx_file": checkpoint_display_path(onnx_path),
        "head": head,
        "num_blocks": args["num_blocks"],
        "seed": args["seed"],
        "input_length": len(TEST_TEXT),
        "batch_size": 1,
        "cpu_threads": 1,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "macs": macs,
        "flops": 2 * macs,
        "onnx_forward_mean_ms": forward["mean_ms"],
        "decode_entities_mean_ms": decoding["mean_ms"],
        "complete_mean_latency_ms": complete["mean_ms"],
        "complete_p50_latency_ms": complete["p50_ms"],
        "complete_p95_latency_ms": complete["p95_ms"],
        "throughput_samples_per_second": 1000.0 / complete["mean_ms"],
        "provider": "CPUExecutionProvider",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    checkpoints = sorted((ROOT / "outputs").glob("*_b*_seed*/best.pt"))
    if len(checkpoints) != 48:
        raise RuntimeError(f"expected 48 checkpoints, found {len(checkpoints)}")

    onnx_dir = ROOT / "outputs" / "onnx"
    rows = []
    for index, path in enumerate(checkpoints, 1):
        row = profile_checkpoint(path, onnx_dir, args.warmup, args.iterations)
        rows.append(row)
        print(
            f"[{index:02d}/48] {row['run_id']} "
            f"forward={row['onnx_forward_mean_ms']:.3f} ms "
            f"complete={row['complete_mean_latency_ms']:.3f} ms"
        )

    output_path = ROOT / "outputs" / "reports" / "model_complexity_64_onnx.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
