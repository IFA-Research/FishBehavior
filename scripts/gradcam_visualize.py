import argparse
import os
import sys

print("gradcam script started", flush=True)

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from pytorch_grad_cam import GradCAMPlusPlus
from pytorch_grad_cam.utils.image import show_cam_on_image
from transformers import MobileViTForImageClassification

sys.stdout.reconfigure(line_buffering=True)


class MobileViT_BiLSTM(nn.Module):
    def __init__(self, num_classes=4, hidden_size=256):
        super().__init__()
        backbone = MobileViTForImageClassification.from_pretrained("apple/mobilevit-small")
        backbone.classifier = nn.Identity()
        self.backbone = backbone
        self.lstm1 = nn.LSTM(640, hidden_size, batch_first=True, bidirectional=True)
        self.lstm2 = nn.LSTM(hidden_size * 2, hidden_size, batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        features = self.backbone(x).logits.view(b, t, -1)
        x, _ = self.lstm1(features)
        x, _ = self.lstm2(x)
        x = self.fc1(x[:, -1, :])
        x = self.dropout(torch.relu(self.bn(x)))
        return self.fc2(x)


def class_names_from(classes):
    return sorted(str(c) for c in classes)


def load_config(path):
    import json
    with open(path, "r") as f:
        return json.load(f)


def get_layer(model, path):
    layer = model
    for p in path.split("."):
        layer = layer[int(p)] if p.isdigit() else getattr(layer, p)
    return layer


def read_labels(csv_path, classes):
    df = pd.read_csv(csv_path, keep_default_na=False, na_values=[""])
    if "Hunger Level" in df.columns:
        hunger = df["Hunger Level"].astype(str).str.strip()
    elif "hunger_level" in df.columns:
        hunger = df["hunger_level"].astype(str).str.strip()
    else:
        return None
    frame_col = "Frame Number" if "Frame Number" in df.columns else "frame_number"
    if frame_col not in df.columns:
        return None
    out = pd.DataFrame({
        "frame": pd.to_numeric(df[frame_col], errors="coerce"),
        "label": hunger,
    })
    out = out[out["label"].isin(classes)]
    out = out.dropna(subset=["frame"])
    out["frame"] = out["frame"].astype(int)
    out = out.drop_duplicates("frame").sort_values("frame").reset_index(drop=True)
    if out.empty:
        return None
    return out


def find_sessions(base_dir, label_csv, image_subdir, first_n_days=None):
    base = Path(base_dir)
    sessions = []
    if not base.exists():
        return sessions
    for csv_path in sorted(base.rglob(label_csv)):
        image_dir = Path(str(csv_path.parent / image_subdir).replace("Feeding_resized", "Feeding"))
        print("image folder:", image_dir, flush=True)
        original = Path(str(image_dir).replace("Feeding_resized", "Feeding"))
        if original.is_dir() and any(original.glob("*.jpg")):
            image_dir = original
        if image_dir.is_dir() and any(image_dir.glob("*.jpg")):
            sessions.append((csv_path.parent, image_dir, csv_path))
    if first_n_days and sessions:
        dates = sorted({time_dir.parent.name for time_dir, _, _ in sessions})
        keep = set(dates[:first_n_days])
        skipped = dates[first_n_days:]
        sessions = [s for s in sessions if s[0].parent.name in keep]
        print(f"FD days kept: {dates[:first_n_days]}", flush=True)
        print(f"FD days skipped: {skipped}", flush=True)
    return sessions


def run_one(config_path, checkpoint_path, base_dir, output_dir, n_per_class=5, display_size=960, first_n_days=None, max_tries=3000):
    print(f"start {output_dir}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    config = load_config(config_path)
    data_cfg = config["data"]
    classes = class_names_from(data_cfg.get("classes", ["Medium", "Strong", "Weak", "None"]))
    print(f"Class index order: {list(enumerate(classes))}", flush=True)
    seq_len = data_cfg.get("seq_len", 10)
    frame_pattern = data_cfg.get("frame_pattern", "frame_{:06d}.jpg")
    image_subdir = data_cfg.get("image_subdir", "left")
    label_csv = data_cfg.get("label_csv_name")
    img_size = int(data_cfg.get("img_height", 224))

    print("loading model", flush=True)
    model = MobileViT_BiLSTM(num_classes=len(classes)).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if isinstance(state, dict) and "model_state" in state:
        state = state["model_state"]
    model.load_state_dict(state)
    model.eval()
    print("model loaded", flush=True)

    target_layer = get_layer(model, "backbone.mobilevit.conv_1x1_exp.convolution")

    class BackboneWrapper(nn.Module):
        def __init__(self, backbone):
            super().__init__()
            self.backbone = backbone

        def forward(self, x):
            out = self.backbone(x)
            return out.logits if hasattr(out, "logits") else out

    wrapped = BackboneWrapper(model.backbone).to(device).eval()

    def load_frame(path):
        image = Image.open(path).convert("RGB")
        image = image.resize((img_size, img_size))
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        array = np.asarray(image).astype(np.float32) / 255.0
        return torch.from_numpy(array).permute(2, 0, 1)

    sessions = find_sessions(base_dir, label_csv, image_subdir, first_n_days=first_n_days)
    print(f"{output_dir}: {len(sessions)} sessions under {base_dir}", flush=True)
    if not sessions:
        return

    candidates = {cls: [] for cls in classes}
    available = {cls: 0 for cls in classes}
    for time_dir, image_dir, csv_path in sessions:
        labels = read_labels(csv_path, classes)
        if labels is None or len(labels) < seq_len:
            continue
        frames = labels["frame"].to_numpy()
        names = labels["label"].tolist()
        for end in range(seq_len - 1, len(frames)):
            cls = names[end]
            if cls not in candidates:
                continue
            available[cls] += 1
            candidates[cls].append((image_dir, time_dir, frames, end))

    print(f"Labeled windows: {available}", flush=True)

    rng = np.random.default_rng(42)
    os.makedirs(output_dir, exist_ok=True)
    saved = {cls: 0 for cls in classes}

    for cls in classes:
        if cls != "Strong":
            continue
        pool = candidates[cls]
        print(f"checking {cls}: {len(pool)} windows, max {max_tries}", flush=True)
        if not pool:
            continue
        order = rng.permutation(len(pool))
        tried = 0
        for pick in order:
            if saved[cls] >= n_per_class or tried >= max_tries:
                break
            tried += 1
            image_dir, time_dir, frames, end = pool[int(pick)]
            window = frames[end - seq_len + 1: end + 1]
            paths = [image_dir / frame_pattern.format(int(fn)) for fn in window]
            if not all(p.exists() for p in paths):
                if tried % 20 == 0:
                    print(f"{output_dir} {cls}: try {tried} saved {saved[cls]} (missing file)", flush=True)
                continue

            tensors = [load_frame(p) for p in paths]
            seq_tensor = torch.stack(tensors).unsqueeze(0).to(device)

            with torch.no_grad():
                logits = model(seq_tensor)
                probs = torch.softmax(logits, dim=1)[0]
                pred = int(probs.argmax().item())
                conf = float(probs[pred].item())

            if tried % 20 == 0 or (classes[pred] == cls and conf >= 0.99):
                print(
                    f"{output_dir} {cls}: try {tried}/{max_tries} saved {saved[cls]} "
                    f"pred={classes[pred]} conf={conf:.1%}",
                    flush=True,
                )
            if classes[pred] != cls or conf < 0.50:
                continue

            last_path = paths[-1]
            original = Image.open(last_path).convert("RGB")
            scale = display_size / max(original.size)
            disp = original.resize(
                (max(1, int(original.size[0] * scale)), max(1, int(original.size[1] * scale))),
                Image.BILINEAR,
            )
            rgb = np.asarray(disp).astype(np.float32) / 255.0
            single = seq_tensor[:, -1]

            cam = GradCAMPlusPlus(model=wrapped, target_layers=[target_layer])
            heat = cam(input_tensor=single, targets=None)[0]
            heat = cv2.resize(heat, (rgb.shape[1], rgb.shape[0]))
            vis = show_cam_on_image(rgb, heat, use_rgb=True, image_weight=0.65)

            date_name = time_dir.parent.name
            time_name = time_dir.name
            out_name = f"{cls}_{date_name}_{time_name}_{image_dir.name}_{last_path.stem}.jpg"
            out_path = Path(output_dir) / out_name
            n = 1
            while out_path.exists():
                out_path = Path(output_dir) / f"{cls}_{date_name}_{time_name}_{image_dir.name}_{last_path.stem}_{n}.jpg"
                n += 1
            cv2.imwrite(str(out_path), cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
            saved[cls] += 1
            print(f"Saved: {out_path} | GT={cls} pred={classes[pred]} conf={conf:.2%} CORRECT", flush=True)

    print(f"Saved counts for {output_dir}: {saved}", flush=True)
    missing = [c for c in classes if saved[c] < n_per_class]
    if missing:
        print(f"Could not find {n_per_class} correct samples for: {missing}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=Path("configs/base.json"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("gradcam"))
    parser.add_argument("--n-per-class", type=int, default=5)
    parser.add_argument("--first-n-days", type=int, default=None)
    args = parser.parse_args()
    run_one(
        config_path=str(args.config),
        checkpoint_path=str(args.checkpoint),
        base_dir=str(args.base_dir),
        output_dir=str(args.output_dir),
        n_per_class=args.n_per_class,
        first_n_days=args.first_n_days,
    )


if __name__ == "__main__":
    main()