"""Train a binary Normal vs Defective ResNet18 classifier."""
from pathlib import Path
import argparse
import copy
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageFile

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

ImageFile.LOAD_TRUNCATED_IMAGES = False

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets" / "dataset_manifest.csv"
OUTPUT = ROOT / "models" / "tablet_classifier"
REPORTS = ROOT / "reports" / "training"


class TabletDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = ROOT / str(row["image_path"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = 1 if row["label"] == "defective" else 0
        return image, torch.tensor(label, dtype=torch.long)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def calculate_metrics(y_true, y_pred):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_defective": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_defective": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "f1_defective": f1_score(
            y_true, y_pred, zero_division=0
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="Train without downloaded ImageNet pretrained weights."
    )
    args = parser.parse_args()

    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    if not MANIFEST.exists():
        raise SystemExit(f"Manifest not found: {MANIFEST}")

    df = pd.read_csv(MANIFEST).fillna("")
    required = {"image_path", "split", "label"}
    if not required.issubset(df.columns):
        raise SystemExit(f"Manifest must contain: {sorted(required)}")

    df = df[df["task"].astype(str).str.strip().str.lower() == "tablet"].copy() if "task" in df.columns else df.copy()

    train_df = df[df["split"] == "train"].copy()
    val_df = df[df["split"] == "validation"].copy()

    for name, frame in [("train", train_df), ("validation", val_df)]:
        if frame.empty:
            raise SystemExit(f"No {name} images found.")
        if not set(frame["label"]).issubset({"normal", "defective"}):
            raise SystemExit(f"Unexpected labels in {name}: {frame['label'].unique()}")
        missing = [str(ROOT / p) for p in frame["image_path"]
                   if not (ROOT / p).is_file()]
        if missing:
            raise SystemExit(f"{len(missing)} missing {name} image files. First: {missing[0]}")

    print("Training images:", len(train_df))
    print("Training labels:\n", train_df["label"].value_counts().to_string())
    print("Validation images:", len(val_df))
    print("Validation labels:\n", val_df["label"].value_counts().to_string())

    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.10, contrast=0.10),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    train_loader = DataLoader(
        TabletDataset(train_df, train_transform),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0
    )
    val_loader = DataLoader(
        TabletDataset(val_df, val_transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0
    )

    weights = None if args.scratch else models.ResNet18_Weights.DEFAULT
    try:
        model = models.resnet18(weights=weights)
    except Exception as exc:
        raise SystemExit(
            "Could not load pretrained ResNet18 weights. Check your internet "
            "connection, or rerun with --scratch. Details: " + str(exc)
        )

    model.fc = nn.Linear(model.fc.in_features, 2)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=0.0001, weight_decay=0.0001
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2
    )

    OUTPUT.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    best_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss_total = 0.0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss_total += loss.item() * images.size(0)

        model.eval()
        val_true, val_pred = [], []
        val_loss_total = 0.0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                loss = criterion(logits, labels)
                val_loss_total += loss.item() * images.size(0)
                predictions = torch.argmax(logits, dim=1)
                val_true.extend(labels.cpu().tolist())
                val_pred.extend(predictions.cpu().tolist())

        metrics = calculate_metrics(val_true, val_pred)
        train_loss = train_loss_total / len(train_df)
        val_loss = val_loss_total / len(val_df)
        scheduler.step(metrics["f1_defective"])

        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_loss": val_loss,
            **metrics,
        }
        history.append(record)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | "
            f"val_accuracy={metrics['accuracy']:.4f} | "
            f"defective_recall={metrics['recall_defective']:.4f} | "
            f"defective_F1={metrics['f1_defective']:.4f}"
        )

        if metrics["f1_defective"] > best_f1:
            best_f1 = metrics["f1_defective"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
            torch.save({
                "model_state_dict": best_state,
                "class_to_index": {"normal": 0, "defective": 1},
                "image_size": 224,
                "architecture": "resnet18",
                "pretrained": not args.scratch,
                "selected_threshold": 0.5,
                "best_validation_metrics": metrics,
                "best_epoch": epoch,
                "seed": 42,
            }, OUTPUT / "best_model.pth")
            print("  Saved new best validation checkpoint.")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= args.patience:
            print("Early stopping: validation F1 did not improve.")
            break

    pd.DataFrame(history).to_csv(
        REPORTS / "training_history.csv", index=False
    )

    print("\nTraining finished.")
    print("Best validation defective F1:", round(best_f1, 4))
    print("Best model:", OUTPUT / "best_model.pth")
    print("Training history:", REPORTS / "training_history.csv")
    print("Test images were not loaded or evaluated.")


if __name__ == "__main__":
    main()
