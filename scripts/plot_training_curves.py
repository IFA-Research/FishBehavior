"""Plot training loss for the four temporal variants.

One figure per camera, epochs 1-100, no title. BiLSTM is drawn thicker.
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

# Folder names that may differ from the config file name.
VARIANTS = [
    ("Mean pooling", ["fd_mean_pooling", "fd_mean"], ["zed_mean"]),
    ("BiLSTM", ["fd_bilstm"], ["zed_bilstm"]),
    ("Temporal Transformer", ["fd_transformer"], ["zed_transformer"]),
    ("BiLSTM + Transformer", ["fd_bilstm_transformer"], ["zed_bilstm_transformer"]),
]
CAMERAS = ("FD", "ZED")
COLORS = {
    "Mean pooling": "#4C78A8",
    "BiLSTM": "#F58518",
    "Temporal Transformer": "#54A24B",
    "BiLSTM + Transformer": "#E45756",
}


def resolve_folder(runs_dir: Path, names: list[str]) -> Path | None:
    found = [runs_dir / name for name in names if (runs_dir / name).is_dir()]
    if not found:
        return None

    def rank(folder: Path):
        has_xlsx = (folder / f"{folder.name}_results.xlsx").exists()
        has_ckpt = (folder / f"{folder.name}_checkpoint.pth").exists()
        return (has_xlsx, has_ckpt)

    found.sort(key=rank, reverse=True)
    return found[0]


def load_history(folder: Path) -> tuple[pd.DataFrame, str]:
    xlsx = folder / f"{folder.name}_results.xlsx"
    if xlsx.exists():
        df = pd.read_excel(xlsx, sheet_name="history")
        return df, xlsx.name

    ckpt = folder / f"{folder.name}_checkpoint.pth"
    if ckpt.exists():
        import torch

        try:
            obj = torch.load(ckpt, map_location="cpu", weights_only=False)
        except TypeError:
            obj = torch.load(ckpt, map_location="cpu")
        if "history" not in obj:
            raise KeyError(f"No history in {ckpt}")
        return pd.DataFrame(obj["history"]), ckpt.name

    raise FileNotFoundError(
        f"No {folder.name}_results.xlsx or {folder.name}_checkpoint.pth in {folder}"
    )


def collect(runs_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    histories = []
    rows = []
    for label, fd_names, zed_names in VARIANTS:
        for camera, names in (("FD", fd_names), ("ZED", zed_names)):
            folder = resolve_folder(runs_dir, names)
            if folder is None:
                print(f"MISSING  {camera:3}  {label}")
                continue
            df, source = load_history(folder)
            required = {"epoch", "train_loss", "val_loss", "val_weighted_f1"}
            missing = required - set(df.columns)
            if missing:
                raise KeyError(f"{folder.name} history is missing columns: {sorted(missing)}")

            df = df.copy()
            df["camera"] = camera
            df["variant"] = label
            df["run"] = folder.name
            histories.append(df)

            best = df.loc[df["val_weighted_f1"].idxmax()]
            rows.append(
                {
                    "camera": camera,
                    "variant": label,
                    "run": folder.name,
                    "source": source,
                    "epochs_ran": int(df["epoch"].max()),
                    "best_epoch": int(best["epoch"]),
                    "best_train_loss": float(best["train_loss"]),
                    "best_val_loss": float(best["val_loss"]),
                    "best_val_accuracy": float(best.get("val_accuracy", float("nan"))),
                    "best_val_weighted_f1": float(best["val_weighted_f1"]),
                    "best_val_macro_f1": float(best.get("val_macro_f1", float("nan"))),
                    "final_train_loss": float(df.iloc[-1]["train_loss"]),
                    "final_val_loss": float(df.iloc[-1]["val_loss"]),
                    "final_val_weighted_f1": float(df.iloc[-1]["val_weighted_f1"]),
                }
            )
            print(
                f"OK       {camera:3}  {label:24}  epochs={int(df['epoch'].max()):3}  "
                f"best_epoch={int(best['epoch']):3}  "
                f"val_wf1={float(best['val_weighted_f1']):.4f}  source={source}"
            )

    if not rows:
        raise SystemExit(f"No runs found in {runs_dir}")
    return pd.concat(histories, ignore_index=True), pd.DataFrame(rows)


def print_ranking(summary: pd.DataFrame) -> None:
    print("\nRanking by best validation weighted F1")
    winners = {}
    for camera in CAMERAS:
        part = summary[summary["camera"] == camera].sort_values(
            "best_val_weighted_f1", ascending=False
        )
        if part.empty:
            print(f"\n{camera}: no runs")
            continue
        print(f"\n{camera}")
        for rank, row in enumerate(part.itertuples(index=False), start=1):
            print(
                f"  {rank}. {row.variant:24}  "
                f"wf1={row.best_val_weighted_f1:.4f}  "
                f"mf1={row.best_val_macro_f1:.4f}  "
                f"acc={row.best_val_accuracy:.4f}  "
                f"epoch={row.best_epoch}"
            )
        winners[camera] = part.iloc[0]["variant"]
    if len(winners) == 2:
        same = winners["FD"] == winners["ZED"]
        print(
            f"\nBest FD: {winners['FD']} | Best ZED: {winners['ZED']} | "
            f"same winner: {'yes' if same else 'no'}"
        )


def plot_losses(history: pd.DataFrame, out_dir: Path) -> None:
    plotted = history[(history["epoch"] <= 100) & (history["variant"] == "BiLSTM")]
    for camera in CAMERAS:
        curve = plotted[plotted["camera"] == camera].sort_values("epoch")
        if curve.empty:
            print(f"MISSING plot {camera}")
            continue
        fig, ax = plt.subplots(figsize=(6.6, 4.2))
        ax.plot(
            curve["epoch"],
            curve["train_loss"],
            color=COLORS["BiLSTM"],
            linewidth=2.0,
        )
        ax.set_xlim(1, 100)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Training loss")
        ax.grid(True, linewidth=0.4, alpha=0.4)
        fig.tight_layout()
        path = out_dir / f"training_loss_{camera.lower()}.png"
        fig.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    history, summary = collect(args.runs)
    print_ranking(summary)
    history.to_csv(args.out / "training_history_long.csv", index=False)
    summary_path = args.out / "training_summary.xlsx"
    summary.sort_values(["camera", "best_val_weighted_f1"], ascending=[True, False]).to_excel(
        summary_path, index=False
    )
    print(f"Saved {summary_path}")
    plot_losses(history, args.out)


if __name__ == "__main__":
    main()