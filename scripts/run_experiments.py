import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", nargs="+", default=["softmax"], choices=["softmax", "crf", "egp"])
    parser.add_argument("--num_blocks", nargs="+", type=int, default=[1, 2, 3, 4], choices=[1, 2, 3, 4])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch_size", type=int)
    parser.add_argument("--max_len", type=int)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    for head in args.heads:
        for num_blocks in args.num_blocks:
            for seed in args.seeds:
                run_id = f"{head}_b{num_blocks}_seed{seed}"
                metrics_path = ROOT / "outputs" / run_id / "metrics.json"
                if metrics_path.exists() and not args.force:
                    print(f"skip existing {run_id}: {metrics_path}", flush=True)
                    continue
                cmd = [
                    sys.executable,
                    "src/train.py",
                    "--head",
                    head,
                    "--num_blocks",
                    str(num_blocks),
                    "--seed",
                    str(seed),
                ]
                if args.epochs is not None:
                    cmd.extend(["--epochs", str(args.epochs)])
                if args.batch_size is not None:
                    cmd.extend(["--batch_size", str(args.batch_size)])
                if args.max_len is not None:
                    cmd.extend(["--max_len", str(args.max_len)])
                if args.cpu:
                    cmd.append("--cpu")
                print(" ".join(cmd), flush=True)
                subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
