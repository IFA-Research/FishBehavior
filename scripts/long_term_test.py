"""Long-term color timelines for the proposed BiLSTM model.

Select a session that contains all four labels inside the first three minutes,
then draw only that interval.
"""
import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageFilter
from matplotlib.patches import Patch
from sklearn.preprocessing import OneHotEncoder
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models.model import FishBehaviorModel

COLORS = {
    "Strong": "#FF6B6B",
    "Medium": "#FFD93D",
    "Weak": "#6BCB77",
    "None": "#A8A8A8",
}
LEGEND_ORDER = ["Strong", "Medium", "Weak", "None"]
PLOT_SECONDS = 180.0


def load_config(name: str) -> dict:
    path = ROOT / "configs" / name
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def class_names_from(classes: list[str]) -> list[str]:
    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(classes).reshape(-1, 1))
    return [str(c) for c in encoder.categories_[0]]


def read_labels(path: Path, classes: list[str]) -> pd.DataFrame | None:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    if "Hunger Level" in df.columns:
        df["hunger_level"] = df["Hunger Level"].astype(str).str.strip()
    elif "hunger_level" in df.columns:
        df["hunger_level"] = df["hunger_level"].astype(str).str.strip()
    else:
        return None
    frame_col = "Frame Number" if "Frame Number" in df.columns else "frame_number"
    if frame_col not in df.columns:
        return None
    df = df[df["hunger_level"].isin(classes)].copy()
    df[frame_col] = pd.to_numeric(df[frame_col], errors="coerce")
    df = df.dropna(subset=[frame_col])
    df[frame_col] = df[frame_col].astype(int)
    df = df.sort_values(frame_col).drop_duplicates(frame_col).reset_index(drop=True)
    if df.empty:
        return None
    df["frame_number"] = df[frame_col]
    df["time_s"] = time_axis(df)
    return df


def time_axis(df: pd.DataFrame) -> np.ndarray:
    for col in ("timestamp_seconds", "seconds"):
        if col in df.columns:
            values = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(values).sum() > 1:
                filled = pd.Series(values).ffill().bfill().to_numpy(dtype=float)
                return filled - np.nanmin(filled)
    if "Date Time" in df.columns:
        stamps = pd.to_datetime(df["Date Time"], errors="coerce")
        if stamps.notna().sum() > 1:
            seconds = (stamps - stamps.min()).dt.total_seconds().to_numpy(dtype=float)
            seconds = pd.Series(seconds).ffill().bfill().to_numpy(dtype=float)
            return seconds - np.nanmin(seconds)
    frames = df["frame_number"].to_numpy(dtype=float)
    return (frames - frames.min()) / 10.0


def list_sessions(data_cfg: dict) -> list[dict]:
    base = Path(data_cfg["base_dir"])
    classes = data_cfg["classes"]
    label_name = data_cfg["label_csv_name"]
    found = []
    for date_dir in sorted(p for p in base.iterdir() if p.is_dir()):
        for time_dir in sorted(p for p in date_dir.iterdir() if p.is_dir()):
            csv_path = time_dir / label_name
            image_dir = time_dir / data_cfg["image_subdir"]
            if not (csv_path.exists() and image_dir.exists()):
                continue
            df = read_labels(csv_path, classes)
            if df is None:
                continue
            df = df[df["time_s"] <= PLOT_SECONDS].reset_index(drop=True)
            if len(df) < data_cfg["seq_len"]:
                continue
            counts = df["hunger_level"].value_counts().to_dict()
            found.append(
                {
                    "session": f"{date_dir.name}/{time_dir.name}",
                    "image_dir": image_dir,
                    "n_rows": len(df),
                    "duration_s": float(df["time_s"].iloc[-1] - df["time_s"].iloc[0]),
                    "n_classes": int(df["hunger_level"].nunique()),
                    "counts": {name: int(counts.get(name, 0)) for name in classes},
                    "labels": df,
                }
            )
    return found


def choose_sessions(sessions: list[dict], requested: str | None, top: int) -> list[dict]:
    if requested:
        for item in sessions:
            if item["session"] == requested:
                return [item]
        known = ", ".join(item["session"] for item in sessions)
        raise SystemExit(f"Session {requested} was not found. Known sessions: {known}")
    full = [item for item in sessions if item["n_classes"] == 4]
    if not full:
        raise SystemExit("No session contains all four labels in the first 3 minutes.")
    full.sort(key=lambda item: (min(item["counts"].values()), item["n_rows"]), reverse=True)
    return full[:top]


def print_candidates(name: str, sessions: list[dict], chosen: str) -> None:
    ranked = sorted(sessions, key=lambda item: (item["n_classes"], min(item["counts"].values())), reverse=True)
    print(f"\n{name} candidates (top 8):")
    for item in ranked[:8]:
        mark = " <--" if item["session"] == chosen else ""
        counts = " ".join(f"{k[0]}={v}" for k, v in item["counts"].items())
        print(f"  {item['session']:22}  classes={item['n_classes']}  {item['duration_s']/60:.1f} min  {counts}{mark}")


def load_model(config: dict, weight_path: Path, device: torch.device) -> torch.nn.Module:
    model = FishBehaviorModel(config).to(device)
    try:
        state = torch.load(weight_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(weight_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    return model


def preprocess(path: Path, size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    image = image.resize((size, size))
    image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    array = np.asarray(image).astype(np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1)


def predict_session(model, session: dict, data_cfg: dict, names: list[str], device, batch_size: int):
    df = session["labels"]
    seq_len = data_cfg["seq_len"]
    step = data_cfg["step"]
    pattern = data_cfg["frame_pattern"]
    image_dir = session["image_dir"]
    frames = df["frame_number"].to_numpy()
    times = df["time_s"].to_numpy(dtype=float)
    labels = df["hunger_level"].tolist()

    windows = []
    for start in range(0, len(frames) - seq_len + 1, step):
        end = start + seq_len - 1
        paths = [image_dir / pattern.format(int(fn)) for fn in frames[start:start + seq_len]]
        if all(path.exists() for path in paths):
            windows.append((paths, times[end], labels[end]))

    pred_times = []
    pred_labels = []
    gt_labels = []
    for offset in tqdm(range(0, len(windows), batch_size), desc=session["session"]):
        batch = windows[offset:offset + batch_size]
        clips = []
        for paths, _, _ in batch:
            clips.append(torch.stack([preprocess(path, data_cfg["img_height"]) for path in paths]))
        x = torch.stack(clips).to(device)
        with torch.no_grad():
            logits = model(x)
        chosen = logits.argmax(dim=1).cpu().numpy()
        for (_, t, gt), index in zip(batch, chosen):
            pred_times.append(float(t))
            pred_labels.append(names[int(index)])
            gt_labels.append(gt)
    return np.asarray(pred_times), pred_labels, gt_labels


def as_segments(times: np.ndarray, labels: list[str]) -> list[tuple[float, float, str]]:
    if len(times) == 0:
        return []
    gaps = np.diff(times)
    step = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 1.0
    raw = []
    for i, label in enumerate(labels):
        end = times[i + 1] if i + 1 < len(times) else times[i] + step
        raw.append((float(times[i]), float(end), label))
    merged = [raw[0]]
    for start, end, label in raw[1:]:
        prev_s, _, prev_l = merged[-1]
        if label == prev_l:
            merged[-1] = (prev_s, end, prev_l)
        else:
            merged.append((start, end, label))
    return merged


def clip_segments(segments, t1: float = PLOT_SECONDS):
    clipped = []
    for start, end, label in segments:
        if start >= t1:
            continue
        clipped.append((max(0.0, start), min(end, t1), label))
    if clipped and clipped[-1][1] < t1:
        start, _, label = clipped[-1]
        clipped[-1] = (start, t1, label)
    return clipped


def plot_ribbon(segments, out_path: Path) -> None:
    segments = clip_segments(segments)
    scale = 60.0
    fig, ax = plt.subplots(figsize=(10.6, 1.55))
    ax.barh(0, PLOT_SECONDS / scale, left=0, height=0.62, color="#E6E6E6")
    for start, end, label in segments:
        ax.barh(
            0,
            (end - start) / scale,
            left=start / scale,
            height=0.62,
            color=COLORS.get(label, "#A8A8A8"),
            linewidth=0,
        )
    ax.set_xlim(0, PLOT_SECONDS / scale)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_ylim(-0.55, 0.7)
    ax.set_yticks([])
    ax.set_xlabel("Time (min)")
    ax.tick_params(axis="x", labelsize=10)
    for side in ("left", "right", "top"):
        ax.spines[side].set_visible(False)
    handles = [Patch(facecolor=COLORS[name], edgecolor="none", label=name) for name in LEGEND_ORDER]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.48),
        ncol=4,
        frameon=False,
        fontsize=9,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out_path}")


def agreement(gt: list[str], pred: list[str]) -> None:
    if not gt:
        print("No predictions.")
        return
    hit = sum(a == b for a, b in zip(gt, pred))
    print(f"Sequence agreement on this session: {hit / len(gt):.4f} ({hit}/{len(gt)})")


def run_dataset(config_path: Path, checkpoint: Path | None, device, out_dir: Path, requested: str | None, batch_size: int, top: int) -> None:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    data_cfg = config["data"]
    tag = config.get("experiment_name", Path(config_path).stem)
    names = class_names_from(data_cfg["classes"])
    sessions = list_sessions(data_cfg)
    if not sessions:
        raise SystemExit(f"No sessions found in {data_cfg['base_dir']}")
    chosen = choose_sessions(sessions, requested, top)
    print_candidates(tag, sessions, chosen[0]["session"])
    for rank, session in enumerate(chosen, start=1):
        print(f"Rank {rank}: {session['session']}  ({session['duration_s'] / 60:.1f} min)  {session['counts']}")

    if checkpoint is None:
        weight = ROOT / "runs" / tag / f"{tag}_best.pth"
        if not weight.exists():
            weight = ROOT / "runs" / tag / f"{tag}_checkpoint.pth"
    else:
        weight = Path(checkpoint)
    print(f"Weights: {weight}")
    model = load_model(config, weight, device)

    for rank, session in enumerate(chosen, start=1):
        print(f"\nRunning rank {rank}: {session['session']}")
        pred_t, pred_y, window_gt = predict_session(model, session, data_cfg, names, device, batch_size)
        keep = pred_t <= PLOT_SECONDS
        pred_t = pred_t[keep]
        pred_y = [y for y, flag in zip(pred_y, keep) if flag]
        window_gt = [y for y, flag in zip(window_gt, keep) if flag]
        agreement(window_gt, pred_y)

        labels = session["labels"]
        gt_segments = as_segments(labels["time_s"].to_numpy(dtype=float), labels["hunger_level"].tolist())
        pred_segments = as_segments(pred_t, pred_y)
        session_tag = session["session"].replace("/", "_")
        stem = f"long_{tag}_{rank}_{session_tag}"
        plot_ribbon(gt_segments, out_dir / f"{stem}_ground_truth.png")
        plot_ribbon(pred_segments, out_dir / f"{stem}_prediction.png")
        pd.DataFrame({"time_s": pred_t, "ground_truth": window_gt, "prediction": pred_y}).to_csv(
            out_dir / f"{stem}_sequences.csv", index=False
        )

    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "base.json")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--session", default=None, help="date/time, for example 2026-07-14/13-06-57")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--top", type=int, default=2, help="How many ranked sessions to draw")
    parser.add_argument("--out", type=Path, default=ROOT / "results" / "long_test")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    run_dataset(args.config, args.checkpoint, device, args.out, args.session, args.batch_size, args.top)


if __name__ == "__main__":
    main()