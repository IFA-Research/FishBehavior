"""Ablation table for the four temporal blocks on FD and ZED.

Order is the design path, not the score order:
  1. Mean pooling                  baseline, no temporal model
  2. BiLSTM                        replace pooling with two LSTM layers
  3. Temporal Transformer          replace the LSTM with one encoder layer
  4. BiLSTM + Transformer          add one encoder layer after the LSTM
"""
import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

ROOT = Path(__file__).resolve().parent.parent

STEPS = [
    {
        "step": 1,
        "variant": "Mean pooling",
        "operation": "baseline",
        "relative_to": "",
        "change": "Average the 640-d frame embeddings. No temporal layer.",
        "fd": ["fd_mean_pooling", "fd_mean"],
        "zed": ["zed_mean"],
    },
    {
        "step": 2,
        "variant": "BiLSTM",
        "operation": "replace",
        "relative_to": "Mean pooling",
        "change": "Replace averaging with two bidirectional LSTM layers. Read the last state.",
        "fd": ["fd_bilstm"],
        "zed": ["zed_bilstm"],
    },
    {
        "step": 3,
        "variant": "Temporal Transformer",
        "operation": "replace",
        "relative_to": "BiLSTM",
        "change": "Replace the LSTM with one Transformer encoder layer. Read the last index.",
        "fd": ["fd_transformer"],
        "zed": ["zed_transformer"],
    },
    {
        "step": 4,
        "variant": "BiLSTM + Transformer",
        "operation": "add",
        "relative_to": "BiLSTM",
        "change": "Keep the two LSTM layers and add one Transformer encoder layer after them.",
        "fd": ["fd_bilstm_transformer"],
        "zed": ["zed_bilstm_transformer"],
    },
]


def resolve_folder(runs_dir: Path, names: list[str]) -> Path | None:
    found = [runs_dir / name for name in names if (runs_dir / name).is_dir()]
    if not found:
        return None

    def rank(folder: Path):
        return (
            (folder / f"{folder.name}_results.xlsx").exists(),
            (folder / f"{folder.name}_checkpoint.pth").exists(),
        )

    found.sort(key=rank, reverse=True)
    return found[0]


def load_history(folder: Path) -> tuple[pd.DataFrame, str]:
    xlsx = folder / f"{folder.name}_results.xlsx"
    if xlsx.exists():
        return pd.read_excel(xlsx, sheet_name="history"), xlsx.name
    ckpt = folder / f"{folder.name}_checkpoint.pth"
    if not ckpt.exists():
        raise FileNotFoundError(folder)
    import torch

    try:
        obj = torch.load(ckpt, map_location="cpu", weights_only=False)
    except TypeError:
        obj = torch.load(ckpt, map_location="cpu")
    return pd.DataFrame(obj["history"]), ckpt.name


def best_metrics(folder: Path) -> dict:
    df, source = load_history(folder)
    best = df.loc[df["val_weighted_f1"].idxmax()]
    return {
        "run": folder.name,
        "source": source,
        "epochs_ran": int(df["epoch"].max()),
        "best_epoch": int(best["epoch"]),
        "best_val_loss": float(best["val_loss"]),
        "best_val_accuracy": float(best["val_accuracy"]),
        "best_val_weighted_f1": float(best["val_weighted_f1"]),
        "best_val_macro_f1": float(best["val_macro_f1"]),
    }


def build(runs_dir: Path) -> pd.DataFrame:
    rows = []
    for camera, key in (("FD", "fd"), ("ZED", "zed")):
        for spec in STEPS:
            folder = resolve_folder(runs_dir, spec[key])
            row = {
                "camera": camera,
                "step": spec["step"],
                "variant": spec["variant"],
                "operation": spec["operation"],
                "relative_to": spec["relative_to"],
                "change": spec["change"],
            }
            if folder is None:
                print(f"MISSING  {camera:3}  {spec['variant']}")
                row["run"] = ""
                rows.append(row)
                continue
            row.update(best_metrics(folder))
            print(
                f"OK       {camera:3}  {spec['variant']:24}  "
                f"wf1={row['best_val_weighted_f1']:.4f}  "
                f"mf1={row['best_val_macro_f1']:.4f}  "
                f"acc={row['best_val_accuracy']:.4f}"
            )
            rows.append(row)

    table = pd.DataFrame(rows)
    for camera in ("FD", "ZED"):
        mask = table["camera"] == camera
        part = table.loc[mask].set_index("variant")
        if "best_val_weighted_f1" not in part.columns or "Mean pooling" not in part.index:
            continue
        mean_wf1 = part.loc["Mean pooling", "best_val_weighted_f1"]
        mean_mf1 = part.loc["Mean pooling", "best_val_macro_f1"]
        bilstm_wf1 = (
            part.loc["BiLSTM", "best_val_weighted_f1"] if "BiLSTM" in part.index else float("nan")
        )
        table.loc[mask, "delta_wf1_vs_mean"] = table.loc[mask, "best_val_weighted_f1"] - mean_wf1
        table.loc[mask, "delta_mf1_vs_mean"] = table.loc[mask, "best_val_macro_f1"] - mean_mf1
        table.loc[mask, "delta_wf1_vs_bilstm"] = table.loc[mask, "best_val_weighted_f1"] - bilstm_wf1
    return table


def write_excel(table: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        table.to_excel(writer, sheet_name="ablation", index=False)
        ws = writer.sheets["ablation"]
        header_fill = PatternFill("solid", fgColor="1F4E79")
        header_font = Font(color="FFFFFF", bold=True)
        thin = Border(
            left=Side(style="thin", color="D9D9D9"),
            right=Side(style="thin", color="D9D9D9"),
            top=Side(style="thin", color="D9D9D9"),
            bottom=Side(style="thin", color="D9D9D9"),
        )
        for cell in ws[1]:
            cell.fill = header_fill
            cell.font = header_font
            cell.alignment = Alignment(wrap_text=True, vertical="center")
        ws.row_dimensions[1].height = 30
        ws.auto_filter.ref = ws.dimensions
        ws.freeze_panes = "A2"

        best_rows = set()
        if "best_val_weighted_f1" in table.columns:
            for camera in table["camera"].unique():
                part = table[table["camera"] == camera]["best_val_weighted_f1"]
                if part.notna().any():
                    best_rows.add(int(part.idxmax()) + 2)

        best_fill = PatternFill("solid", fgColor="E2EFDA")
        for row_idx in range(2, ws.max_row + 1):
            for cell in ws[row_idx]:
                cell.border = thin
                cell.alignment = Alignment(vertical="center", wrap_text=(cell.column == 6))
            if row_idx in best_rows:
                for cell in ws[row_idx]:
                    cell.fill = best_fill

        for idx, column in enumerate(ws.columns, start=1):
            letter = get_column_letter(idx)
            width = 18
            header = str(column[0].value or "")
            if header == "change":
                width = 78
            elif header in {"variant", "relative_to", "run", "source"}:
                width = 28
            ws.column_dimensions[letter].width = width
        for row_idx in range(2, ws.max_row + 1):
            for cell in ws[row_idx]:
                if isinstance(cell.value, float):
                    cell.number_format = "0.0000"
    print(f"Saved {path}")


def plot_bars(table: pd.DataFrame, path: Path) -> None:
    metric = "best_val_weighted_f1"
    if metric not in table.columns or table[metric].notna().sum() == 0:
        print("Skip bar chart: no scores")
        return
    variants = [spec["variant"] for spec in STEPS]
    x = range(len(variants))
    width = 0.36
    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    for offset, camera in ((-width / 2, "FD"), (width / 2, "ZED")):
        part = table[table["camera"] == camera].set_index("variant")
        values = [part.loc[name, metric] if name in part.index else float("nan") for name in variants]
        bars = ax.bar(
            [i + offset for i in x],
            values,
            width=width,
            label=camera,
            color=["#4C78A8", "#F58518"][camera == "ZED"],
        )
        for bar, value in zip(bars, values):
            if value == value:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.005,
                    f"{value:.3f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                )
    ax.set_xticks(list(x))
    ax.set_xticklabels(variants)
    ax.set_ylabel("Best validation weighted F1")
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", linewidth=0.4, alpha=0.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"Saved {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=ROOT / "runs")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    table = build(args.runs)
    csv_path = args.out / "ablation_table.csv"
    table.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")
    write_excel(table, args.out / "ablation_table.xlsx")
    plot_bars(table, args.out / "ablation_weighted_f1.png")

    print("\nDeltas are absolute differences on the 0-1 F1 scale.")
    print("Positive delta_wf1_vs_mean means better than mean pooling.")
    print("Positive delta_wf1_vs_bilstm means better than the two-layer BiLSTM.")


if __name__ == "__main__":
    main()