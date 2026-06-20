from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

HEAD_ORDER = ["softmax", "crf", "egp", "cascade"]
HEAD_LABELS = {
    "softmax": "Softmax",
    "crf": "CRF",
    "egp": "EGP",
    "cascade": "Cascade",
}
COLORS = {
    "softmax": "#4C72B0",
    "crf": "#C44E52",
    "egp": "#55A868",
    "cascade": "#8172B2",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": "#D8DDE6",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, name: str, figures_dir: Path) -> None:
    figures_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ["png", "pdf", "svg"]:
        fig.savefig(figures_dir / f"{name}.{suffix}", bbox_inches="tight")
    plt.close(fig)


def score_limits(values: pd.Series, margin: float = 0.02) -> tuple[float, float]:
    lower = max(0.0, float(values.min()) - margin)
    upper = min(1.0, float(values.max()) + margin)
    return lower, upper


def load_data(
    outputs_dir: Path,
    heads: list[str],
    tag: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_dir = outputs_dir / "reports" / tag
    summary_path = report_dir / "summary.csv"
    mean_std_path = report_dir / "summary_mean_std.csv"
    if not summary_path.exists() or not mean_std_path.exists():
        raise FileNotFoundError("Run scripts/summarize_results.py before plotting.")

    summary = pd.read_csv(summary_path)
    mean_std = pd.read_csv(mean_std_path)
    summary = summary[summary["head"].isin(heads)].copy()
    mean_std = mean_std[mean_std["head"].isin(heads)].copy()
    summary["head"] = pd.Categorical(summary["head"], categories=heads, ordered=True)
    mean_std["head"] = pd.Categorical(mean_std["head"], categories=heads, ordered=True)
    summary = summary.sort_values(["head", "num_blocks", "seed"]).reset_index(drop=True)
    mean_std = mean_std.sort_values(["head", "num_blocks"]).reset_index(drop=True)

    metric_stats = (
        summary.groupby(["head", "num_blocks"], observed=True)
        .agg(
            test_precision_mean=("test_precision", "mean"),
            test_precision_std=("test_precision", "std"),
            test_recall_mean=("test_recall", "mean"),
            test_recall_std=("test_recall", "std"),
            test_f1_mean=("test_f1", "mean"),
            test_f1_std=("test_f1", "std"),
            dev_f1_mean=("dev_f1", "mean"),
            dev_f1_std=("dev_f1", "std"),
            best_epoch_mean=("best_epoch", "mean"),
        )
        .reset_index()
        .fillna(0.0)
    )
    return summary, mean_std, metric_stats


def plot_test_f1_line(
    mean_std: pd.DataFrame,
    heads: list[str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    for head in heads:
        data = mean_std[mean_std["head"] == head]
        ax.errorbar(
            data["num_blocks"],
            data["test_f1_mean"],
            yerr=data["test_f1_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            color=COLORS[head],
            label=HEAD_LABELS[head],
        )
    ax.set_title("Test F1 across IDCNN Blocks")
    ax.set_xlabel("Number of IDCNN blocks")
    ax.set_ylabel("Test F1")
    ax.set_xticks([1, 2, 3, 4])
    ax.set_ylim(*score_limits(mean_std["test_f1_mean"]))
    ax.legend(ncol=len(heads), loc="upper right")
    save_figure(fig, "test_f1_line_errorbar", figures_dir)


def plot_test_f1_grouped_bar(
    mean_std: pd.DataFrame,
    heads: list[str],
    figures_dir: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 4.4))
    x = np.arange(1, 5)
    width = 0.8 / len(heads)
    offsets = {
        head: (index - (len(heads) - 1) / 2) * width
        for index, head in enumerate(heads)
    }
    for head in heads:
        data = mean_std[mean_std["head"] == head].set_index("num_blocks").loc[x]
        ax.bar(
            x + offsets[head],
            data["test_f1_mean"],
            width,
            yerr=data["test_f1_std"],
            capsize=3,
            color=COLORS[head],
            label=HEAD_LABELS[head],
            edgecolor="white",
            linewidth=0.7,
        )
    ax.set_title("Grouped Comparison of Test F1")
    ax.set_xlabel("Number of IDCNN blocks")
    ax.set_ylabel("Test F1")
    ax.set_xticks(x)
    ax.set_ylim(*score_limits(mean_std["test_f1_mean"]))
    ax.legend(ncol=len(heads), loc="upper right")
    save_figure(fig, "test_f1_grouped_bar", figures_dir)


def plot_dev_test_comparison(
    mean_std: pd.DataFrame,
    heads: list[str],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), sharey=True)
    for ax, split in zip(axes, ["dev", "test"], strict=True):
        for head in heads:
            data = mean_std[mean_std["head"] == head]
            ax.errorbar(
                data["num_blocks"],
                data[f"{split}_f1_mean"],
                yerr=data[f"{split}_f1_std"],
                marker="o",
                linewidth=2,
                capsize=3,
                color=COLORS[head],
                label=HEAD_LABELS[head],
            )
        ax.set_title(f"{split.capitalize()} F1")
        ax.set_xlabel("Number of IDCNN blocks")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(*score_limits(pd.concat([mean_std["dev_f1_mean"], mean_std["test_f1_mean"]])))
    axes[0].set_ylabel("F1")
    axes[1].legend(ncol=1, loc="upper right")
    fig.suptitle("Development vs. Test Performance", y=1.03, fontsize=13)
    save_figure(fig, "dev_test_f1_comparison", figures_dir)


def plot_precision_recall_f1(
    metric_stats: pd.DataFrame,
    heads: list[str],
    figures_dir: Path,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.8), sharey=True)
    metrics = [
        ("test_precision", "Precision"),
        ("test_recall", "Recall"),
        ("test_f1", "F1"),
    ]
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        for head in heads:
            data = metric_stats[metric_stats["head"] == head]
            ax.errorbar(
                data["num_blocks"],
                data[f"{metric}_mean"],
                yerr=data[f"{metric}_std"],
                marker="o",
                linewidth=2,
                capsize=3,
                color=COLORS[head],
                label=HEAD_LABELS[head],
            )
        ax.set_title(label)
        ax.set_xlabel("Number of IDCNN blocks")
        ax.set_xticks([1, 2, 3, 4])
        values = pd.concat(
            [
                metric_stats["test_precision_mean"],
                metric_stats["test_recall_mean"],
                metric_stats["test_f1_mean"],
            ]
        )
        ax.set_ylim(*score_limits(values))
    axes[0].set_ylabel("Test score")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Test Precision, Recall, and F1", y=1.04, fontsize=13)
    save_figure(fig, "test_precision_recall_f1", figures_dir)


def plot_heatmap(
    mean_std: pd.DataFrame,
    heads: list[str],
    figures_dir: Path,
) -> None:
    matrix = (
        mean_std.pivot(index="head", columns="num_blocks", values="test_f1_mean")
        .loc[heads, [1, 2, 3, 4]]
        .to_numpy()
    )
    fig, ax = plt.subplots(figsize=(6.2, 2.0 + 0.6 * len(heads)))
    im = ax.imshow(
        matrix,
        cmap="YlGnBu",
        vmin=float(matrix.min()),
        vmax=float(matrix.max()),
        aspect="auto",
    )
    ax.set_title("Test F1 Heatmap")
    ax.set_xlabel("Number of IDCNN blocks")
    ax.set_ylabel("Output head")
    ax.set_xticks(np.arange(4), labels=[1, 2, 3, 4])
    ax.set_yticks(np.arange(len(heads)), labels=[HEAD_LABELS[h] for h in heads])
    midpoint = (float(matrix.min()) + float(matrix.max())) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            color = "white" if matrix[i, j] > midpoint else "#1F2937"
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=9)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Test F1")
    ax.grid(False)
    save_figure(fig, "test_f1_heatmap", figures_dir)


def plot_ranked_models(mean_std: pd.DataFrame, figures_dir: Path) -> None:
    data = mean_std.copy()
    data["label"] = data["head"].astype(str).map(HEAD_LABELS) + " / b=" + data["num_blocks"].astype(str)
    data = data.sort_values("test_f1_mean", ascending=True)
    colors = [COLORS[h] for h in data["head"].astype(str)]

    fig, ax = plt.subplots(figsize=(6.8, 5.0))
    ax.barh(
        data["label"],
        data["test_f1_mean"],
        xerr=data["test_f1_std"],
        color=colors,
        capsize=3,
        edgecolor="white",
        linewidth=0.7,
    )
    ax.set_title("Ranking of Head and Block Combinations")
    ax.set_xlabel("Test F1")
    ax.set_xlim(*score_limits(data["test_f1_mean"]))
    for value, label in zip(data["test_f1_mean"], data["label"], strict=True):
        ax.text(value + 0.002, label, f"{value:.3f}", va="center", fontsize=8)
    save_figure(fig, "test_f1_ranking", figures_dir)


def plot_seed_variation(
    summary: pd.DataFrame,
    heads: list[str],
    figures_dir: Path,
) -> None:
    fig, axes_grid = plt.subplots(
        1,
        len(heads),
        figsize=(4.0 * len(heads), 3.7),
        sharey=True,
        squeeze=False,
    )
    axes = axes_grid[0]
    for ax, head in zip(axes, heads, strict=True):
        data = summary[summary["head"] == head]
        for seed, seed_data in data.groupby("seed"):
            ax.plot(
                seed_data["num_blocks"],
                seed_data["test_f1"],
                marker="o",
                linewidth=1.6,
                label=f"seed {seed}",
            )
        ax.set_title(HEAD_LABELS[head])
        ax.set_xlabel("Number of IDCNN blocks")
        ax.set_xticks([1, 2, 3, 4])
        ax.set_ylim(*score_limits(summary["test_f1"]))
    axes[0].set_ylabel("Test F1")
    axes[-1].legend(loc="lower left")
    fig.suptitle("Seed-level Test F1 Variation", y=1.04, fontsize=13)
    save_figure(fig, "seed_variation_test_f1", figures_dir)


def generate_figures(
    outputs_dir: str | Path,
    heads: list[str],
    tag: str,
) -> Path:
    outputs_dir = Path(outputs_dir)
    figures_dir = outputs_dir / "reports" / tag / "figures"
    configure_style()
    summary, mean_std, metric_stats = load_data(outputs_dir, heads, tag)
    plot_test_f1_line(mean_std, heads, figures_dir)
    plot_test_f1_grouped_bar(mean_std, heads, figures_dir)
    plot_dev_test_comparison(mean_std, heads, figures_dir)
    plot_precision_recall_f1(metric_stats, heads, figures_dir)
    plot_heatmap(mean_std, heads, figures_dir)
    plot_ranked_models(mean_std, figures_dir)
    plot_seed_variation(summary, heads, figures_dir)
    print(f"wrote figures to {figures_dir}")
    return figures_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", default="outputs")
    parser.add_argument("--heads", nargs="+", choices=HEAD_ORDER, default=HEAD_ORDER)
    parser.add_argument("--tag", default="all_heads")
    args = parser.parse_args()

    outputs_dir = Path(args.outputs_dir)
    if not outputs_dir.is_absolute():
        outputs_dir = ROOT / outputs_dir
    generate_figures(outputs_dir, args.heads, args.tag)


if __name__ == "__main__":
    main()
