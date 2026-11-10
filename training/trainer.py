import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassConfusionMatrix
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from tqdm import tqdm
import pandas as pd

def train_model(model, train_paths, train_labels, val_paths, val_labels, config, device):
    data_cfg = config["data"]
    train_cfg = config["train"]
    exp_name = config.get("experiment_name", "run")
    num_classes = data_cfg.get("num_classes", 4)
    class_names = data_cfg.get("classes", ["Medium", "Strong", "Weak", "None"])

    save_dir = f"./runs/{exp_name}"
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"{exp_name}_checkpoint.pth")

    from data.dataset import FishSequenceDataset
    train_ds = FishSequenceDataset(
        train_paths, train_labels,
        img_size=data_cfg["img_height"],
        augment=True,
        augmentations=config.get("augmentations", {})
    )
    val_ds = FishSequenceDataset(val_paths, val_labels, img_size=data_cfg["img_height"], augment=False)

    train_loader = DataLoader(train_ds, batch_size=data_cfg["batch_size"], shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=data_cfg["batch_size"], shuffle=False, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg["lr"], weight_decay=train_cfg["weight_decay"])

    use_class_weight = train_cfg.get("use_class_weight", True)
    if use_class_weight:
        y_train = np.argmax(train_labels, axis=1)
        class_weights = compute_class_weight("balanced", classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
        criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"], weight=class_weights)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=train_cfg["label_smoothing"])

    accuracy_metric = MulticlassAccuracy(num_classes=num_classes, average="micro").to(device)
    weighted_f1 = MulticlassF1Score(num_classes=num_classes, average="weighted").to(device)
    macro_f1 = MulticlassF1Score(num_classes=num_classes, average="macro").to(device)
    confmat_metric = MulticlassConfusionMatrix(num_classes=num_classes).to(device)

    start_epoch = 0
    best_val_f1 = 0.0
    history = []
    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_val_f1 = ckpt["best_val_f1"]
        history = ckpt["history"]
        print(f"Resumed from epoch {start_epoch}")

    for epoch in range(start_epoch, train_cfg["epochs"]):
        model.train()
        running_loss = 0.0
        accuracy_metric.reset()
        weighted_f1.reset()
        macro_f1.reset()
        optimizer.zero_grad()

        for step, (images, labels) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{train_cfg['epochs']}")):
            # Handle multi-augmentation: images may be (B, N_aug, T, C, H, W)
            if images.dim() == 6:
                B, N, T, C, H, W = images.shape
                images = images.view(B * N, T, C, H, W)
                labels = labels.view(B * N, -1)

            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss = loss / train_cfg["accumulation_steps"]
            loss.backward()
            if (step + 1) % train_cfg["accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
            running_loss += loss.item() * train_cfg["accumulation_steps"]
            preds = outputs.argmax(dim=1)
            targets = labels.argmax(dim=1)
            accuracy_metric.update(preds, targets)
            weighted_f1.update(preds, targets)
            macro_f1.update(preds, targets)

        train_acc = accuracy_metric.compute().item()
        train_wf1 = weighted_f1.compute().item()
        train_mf1 = macro_f1.compute().item()
        train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        accuracy_metric.reset()
        weighted_f1.reset()
        macro_f1.reset()
        confmat_metric.reset()
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                val_loss += criterion(outputs, labels.argmax(dim=1)).item()
                preds = outputs.argmax(dim=1)
                targets = labels.argmax(dim=1)
                accuracy_metric.update(preds, targets)
                weighted_f1.update(preds, targets)
                macro_f1.update(preds, targets)
                confmat_metric.update(preds, targets)

        val_acc = accuracy_metric.compute().item()
        val_wf1 = weighted_f1.compute().item()
        val_mf1 = macro_f1.compute().item()
        val_loss = val_loss / len(val_loader)

        print(f"Epoch {epoch+1} | train_loss: {train_loss:.4f} acc: {train_acc:.4f} wf1: {train_wf1:.4f} mf1: {train_mf1:.4f} | "
              f"val_loss: {val_loss:.4f} val_acc: {val_acc:.4f} val_wf1: {val_wf1:.4f} val_mf1: {val_mf1:.4f}")

        history.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "train_weighted_f1": train_wf1,
            "train_macro_f1": train_mf1,
            "val_loss": val_loss,
            "val_accuracy": val_acc,
            "val_weighted_f1": val_wf1,
            "val_macro_f1": val_mf1
        })

        torch.save({
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "history": history
        }, checkpoint_path)

        if val_wf1 > best_val_f1:
            best_val_f1 = val_wf1
            torch.save(model.state_dict(), os.path.join(save_dir, f"{exp_name}_best.pth"))
            print("Best model saved!")

    confmat = confmat_metric.compute().cpu().numpy()
    confmat_df = pd.DataFrame(confmat, index=class_names, columns=class_names)

    best_row = max(history, key=lambda x: x["val_weighted_f1"])
    summary = pd.DataFrame([{
        "experiment": exp_name,
        "best_epoch": best_row["epoch"],
        "best_val_weighted_f1": best_row["val_weighted_f1"],
        "best_val_macro_f1": best_row["val_macro_f1"],
        "best_val_accuracy": best_row["val_accuracy"],
        "final_train_weighted_f1": history[-1]["train_weighted_f1"],
        "final_val_weighted_f1": history[-1]["val_weighted_f1"]
    }])

    excel_path = os.path.join(save_dir, f"{exp_name}_results.xlsx")
    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
        pd.DataFrame(history).to_excel(writer, sheet_name="history", index=False)
        confmat_df.to_excel(writer, sheet_name="confusion_matrix")
        summary.to_excel(writer, sheet_name="summary", index=False)

    torch.save(model.state_dict(), os.path.join(save_dir, f"{exp_name}_final.pth"))
    print(f"Training finished. Results saved to {save_dir}")
