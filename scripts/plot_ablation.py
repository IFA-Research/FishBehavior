"""Bar chart of the ablation table. One figure, two cameras, no title."""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Acc., W-F1, M-F1 — same order and values as the table.
SCORES = {
    "Tilapia-ZED": {
        "Mean pooling": (87.79, 87.88, 86.46),
        "Temporal Transformer": (63.55, 60.26, 49.29),
        "BiLSTM + Transformer": (60.46, 57.43, 47.96),
        "BiLSTM": (93.44, 93.43, 91.53),
        "BiLSTM + augmentation": (96.45, 96.46, 95.85),
    },
    "Tilapia-FD": {
        "Mean pooling": (93.32, 93.34, 94.01),
        "Temporal Transformer": (69.96, 70.13, 69.61),
        "BiLSTM + Transformer": (57.27, 53.25, 49.22),
        "BiLSTM": (97.74, 97.75, 97.78),
        "BiLSTM + augmentation": (98.79, 98.79, 98.78),
    },
}
METRICS = ("Acc.", "W-F1", "M-F1")
COLORS = ("#4C78A8", "#F58518", "#54A24B")


def plot(out_path: Path) -> None:
    settings = list(next(iter(SCORES.values())))
    x = np.arange(len(settings))
    width = 0.24
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4), sharey=True)
    handles = []

    for ax, (camera, rows) in zip(axes, SCORES.items()):
        values = np.array([rows[name] for name in settings])
        for i, (metric, color) in enumerate(zip(METRICS, COLORS)):
            bars = ax.bar(
                x + (i - 1) * width,
                values[:, i],
                width,
                label=metric,
                color=color,
            )
            bars[-1].set_edgecolor("black")
            bars[-1].set_linewidth(1.1)
            if ax is axes[0]:
                handles.append(bars)
        ax.set_xticks(x)
        ax.set_xticklabels(settings, rotation=22, ha="right")
        ax.set_ylim(40, 100)
        ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
        ax.set_axisbelow(True)
        ax.text(0.02, 0.98, camera, transform=ax.transAxes, va="top", ha="left")

    axes[0].set_ylabel("Validation score (%)")
    fig.legend(
        handles,
        METRICS,
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("ablation_scores.png"))
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    plot(args.out)


if __name__ == "__main__":
    main()