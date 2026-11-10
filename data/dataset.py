import os
import numpy as np
import pandas as pd
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

class FishSequenceDataset(Dataset):
    def __init__(self, paths_list, labels, img_size=224, augment=False, augmentations=None):
        self.paths_list = paths_list
        self.labels = labels
        self.img_size = img_size
        self.augment = augment
        self.augmentations = augmentations or {}

    def __len__(self):
        return len(self.paths_list)

    def load_and_process_image(self, path):
        image = Image.open(path).convert("RGB")
        image = image.resize((self.img_size, self.img_size))
        image = image.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
        image = np.array(image).astype(np.float32) / 255.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        return image

    def apply_augmentations(self, images):
        # images shape: (T, C, H, W)
        aug = self.augmentations
        versions = [images]

        if aug.get("Flip", False):
            versions.append(torch.flip(images, dims=[3]))

        if aug.get("Noise", False):
            noise = torch.randn_like(images) * aug.get("Noise_Variance", 0.01)
            noisy = torch.clamp(images + noise, 0, 1)
            versions.append(noisy)

        if aug.get("Random_Crop", False):
            crop_h = int(images.shape[2] * 0.75)
            crop_w = int(images.shape[3] * 0.75)
            top = torch.randint(0, images.shape[2] - crop_h + 1, (1,)).item()
            left = torch.randint(0, images.shape[3] - crop_w + 1, (1,)).item()
            cropped = images[:, :, top:top + crop_h, left:left + crop_w]
            cropped = F.interpolate(cropped, size=(images.shape[2], images.shape[3]), mode="bilinear", align_corners=False)
            versions.append(cropped)

        if aug.get("Gaussian_Blur", False):
            kernel_size = 3
            sigma = 1.0
            ax = torch.arange(-kernel_size // 2 + 1., kernel_size // 2 + 1.)
            xx, yy = torch.meshgrid(ax, ax, indexing="ij")
            kernel = torch.exp(-(xx**2 + yy**2) / (2. * sigma**2))
            kernel = kernel / torch.sum(kernel)
            kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(images.shape[1], 1, 1, 1)
            gaussian_blurred = F.conv2d(images, kernel, groups=images.shape[1], padding=kernel_size // 2)
            versions.append(gaussian_blurred)

        if aug.get("Random_Masking", False):
            masked = images.clone()
            mask_h, mask_w = aug.get("Masking_Size", [32, 32])
            num_masks = int(aug.get("Masking_Per_Image", 4))
            for _ in range(num_masks):
                h, w = images.shape[2], images.shape[3]
                top = torch.randint(0, max(1, h - mask_h), (1,)).item()
                left = torch.randint(0, max(1, w - mask_w), (1,)).item()
                masked[:, :, top:top + mask_h, left:left + mask_w] = 0
            versions.append(masked)

        return torch.stack(versions)  # (N, T, C, H, W)

    def __getitem__(self, idx):
        paths = self.paths_list[idx]
        label = self.labels[idx]
        images = torch.stack([self.load_and_process_image(p) for p in paths])  # (T, C, H, W)

        if self.augment and self.augmentations:
            images = self.apply_augmentations(images)
            label = np.tile(label, (images.shape[0], 1))
        return images, label

def create_sequences(data_cfg):
    base_dir = Path(data_cfg["base_dir"])
    seq_len = data_cfg["seq_len"]
    step = data_cfg["step"]
    classes = data_cfg["classes"]
    label_csv = data_cfg.get("label_csv_name", "Labeled_WaterParameter_new.csv")
    image_subdir = data_cfg.get("image_subdir", "left")
    frame_pattern = data_cfg.get("frame_pattern", "frame_{:06d}.jpg")

    encoder = OneHotEncoder(sparse_output=False)
    encoder.fit(np.array(classes).reshape(-1, 1))

    all_seq_paths = []
    all_seq_labels = []

    for date_folder in sorted(base_dir.iterdir()):
        if not date_folder.is_dir():
            continue
        for time_folder in sorted(date_folder.iterdir()):
            if not time_folder.is_dir():
                continue
            label_file = time_folder / label_csv
            image_dir = time_folder / image_subdir
            if not (label_file.exists() and image_dir.exists()):
                continue

            labels_df = pd.read_csv(label_file, keep_default_na=False)
            if "Hunger Level" in labels_df.columns:
                labels_df["hunger_level"] = labels_df["Hunger Level"].astype(str).str.strip()
            elif "hunger_level" in labels_df.columns:
                labels_df["hunger_level"] = labels_df["hunger_level"].astype(str).str.strip()
            else:
                continue

            frame_col = "Frame Number" if "Frame Number" in labels_df.columns else "frame_number"
            if frame_col not in labels_df.columns:
                continue

            labels_df = labels_df[labels_df["hunger_level"].isin(classes)]
            labels_df[frame_col] = pd.to_numeric(labels_df[frame_col], errors="coerce")
            labels_df = labels_df.dropna(subset=[frame_col])
            labels_df[frame_col] = labels_df[frame_col].astype(int)

            if len(labels_df) < seq_len:
                continue

            encoded = encoder.transform(labels_df["hunger_level"].to_numpy().reshape(-1, 1))
            frame_numbers = labels_df[frame_col].values

            for i in range(0, len(frame_numbers) - seq_len + 1, step):
                seq_f = frame_numbers[i:i + seq_len]
                seq_p = [str(image_dir / frame_pattern.format(int(fn))) for fn in seq_f]
                if all(os.path.exists(p) for p in seq_p):
                    all_seq_paths.append(seq_p)
                    all_seq_labels.append(encoded[i + seq_len - 1])

    return all_seq_paths, np.array(all_seq_labels), encoder

def prepare_data(config):
    data_cfg = config["data"]
    paths, labels, encoder = create_sequences(data_cfg)
    if len(paths) == 0:
        raise ValueError("No sequences found. Check base_dir and data structure.")

    label_indices = np.argmax(labels, axis=1)
    train_idx, val_idx = train_test_split(
        np.arange(len(paths)),
        test_size=data_cfg.get("validation_split", 0.2),
        stratify=label_indices,
        random_state=42
    )

    train_paths = [paths[i] for i in train_idx]
    train_labels = labels[train_idx]
    val_paths = [paths[i] for i in val_idx]
    val_labels = labels[val_idx]

    print(f"Found sessions. Total sequences: {len(paths)}")
    print(f"Train: {len(train_paths)} | Val: {len(val_paths)}")
    return train_paths, train_labels, val_paths, val_labels
