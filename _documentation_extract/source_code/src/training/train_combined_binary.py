from pathlib import Path
from copy import deepcopy
import hashlib
import json
import random

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(".")
MANIFEST = ROOT / "datasets/combined_binary_manifest.csv"

MODEL_DIR = ROOT / "models/combined_binary"
REPORT_DIR = ROOT / "reports/training/combined_binary"

CHECKPOINT = MODEL_DIR / "best_model.pth"

IMAGE_SIZE = 224
BATCH_SIZE = 16
MAX_EPOCHS = 25
PATIENCE = 5
LEARNING_RATE = 0.0001
SEED = 42

CLASS_NAMES = ["normal", "defective"]
CLASS_TO_INDEX = {"normal": 0, "defective": 1}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# DATA VALIDATION
# ============================================================

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

required_columns = [
    "image_path", "label", "split", "source", "defect_type"
]

missing_columns = [
    c for c in required_columns if c not in df.columns
]

if missing_columns:
    raise ValueError(f"Missing manifest columns: {missing_columns}")

if len(df) != 970:
    raise ValueError(f"Expected 970 images, found {len(df)}")

if not df["label"].isin(CLASS_TO_INDEX).all():
    raise ValueError("Unexpected binary label.")

if not df["split"].isin(["train", "validation", "test"]).all():
    raise ValueError("Invalid split found.")

# Verify that identical file contents do not cross splits.
print("Checking exact image duplicates...")

hash_locations = {}

for row in df.itertuples(index=False):
    path = Path(row.image_path)

    if not path.is_file():
        raise FileNotFoundError(f"Image missing: {path}")

    digest = hashlib.sha256(path.read_bytes()).hexdigest()

    if digest in hash_locations:
        previous = hash_locations[digest]

        raise ValueError(
            "Exact duplicate image content detected!\n"
            f"First: {previous}\n"
            f"Second: {row.image_path}\n"
            "Review these records before training."
        )

    hash_locations[digest] = (
        row.image_path, row.source, row.split
    )

print("Exact duplicate content: 0")


# ============================================================
# DATA TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.RandomResizedCrop(
        IMAGE_SIZE, scale=(0.90, 1.0)
    ),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=8),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.05,
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# ============================================================
# DATASET
# ============================================================

class TabletDataset(Dataset):

    def __init__(self, dataframe, transform):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]

        image_path = Path(row["image_path"])

        with Image.open(image_path) as source_image:
            image = ImageOps.exif_transpose(source_image)
            image = image.convert("RGB")

        image = self.transform(image)

        label = CLASS_TO_INDEX[row["label"]]

        return (
            image,
            torch.tensor(label, dtype=torch.long),
            row["image_path"],
        )


train_df = df[df["split"] == "train"].copy()
val_df = df[df["split"] == "validation"].copy()
test_df = df[df["split"] == "test"].copy()

train_loader = DataLoader(
    TabletDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
)

val_loader = DataLoader(
    TabletDataset(val_df, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    TabletDataset(test_df, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained ResNet18...")

model = models.resnet18(
    weights=models.ResNet18_Weights.DEFAULT
)

model.fc = nn.Linear(model.fc.in_features, 2)
model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001,
)


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):
    model.eval()

    actual = []
    predicted = []
    probabilities = []
    paths = []

    with torch.no_grad():
        for images, labels, image_paths in loader:
            images = images.to(DEVICE)

            logits = model(images)
            probs = torch.softmax(logits, dim=1)

            predictions = torch.argmax(probs, dim=1)

            actual.extend(labels.numpy().tolist())
            predicted.extend(predictions.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            paths.extend(list(image_paths))

    return (
        np.array(actual),
        np.array(predicted),
        np.array(probabilities),
        paths,
    )


def calculate_metrics(actual, predicted):
    matrix = confusion_matrix(
        actual, predicted, labels=[0, 1]
    )

    tn, fp, fn, tp = matrix.ravel()

    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0.0

    return {
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(
            balanced_accuracy_score(actual, predicted)
        ),
        "macro_f1": float(
            f1_score(actual, predicted, average="macro", zero_division=0)
        ),
        "defective_precision": float(
            precision_score(actual, predicted, pos_label=1, zero_division=0)
        ),
        "defective_recall": float(
            recall_score(actual, predicted, pos_label=1, zero_division=0)
        ),
        "normal_specificity": float(specificity),
        "defective_false_negative_rate": float(false_negative_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


# ============================================================
# TRAINING
# ============================================================

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

best_macro_f1 = -1.0
best_epoch = 0
epochs_without_improvement = 0
best_state = None
history = []

print("\n========== TRAINING CONFIGURATION ==========")
print("Device:", DEVICE)
print("Training images:", len(train_df))
print("Validation images:", len(val_df))
print("Test images:", len(test_df))
print("Classes:", CLASS_NAMES)

print("\nTraining labels:")
print(train_df["label"].value_counts().to_string())

print("\nStarting training...\n")

for epoch in range(1, MAX_EPOCHS + 1):

    model.train()
    running_loss = 0.0
    total = 0
    correct = 0

    for images, labels, _ in train_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(logits, dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    val_actual, val_predicted, _, _ = evaluate(
        model, val_loader
    )

    val_metrics = calculate_metrics(
        val_actual, val_predicted
    )

    val_f1 = val_metrics["macro_f1"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "train_accuracy": train_accuracy,
        "validation_accuracy": val_metrics["accuracy"],
        "validation_balanced_accuracy": val_metrics["balanced_accuracy"],
        "validation_macro_f1": val_f1,
    })

    print(
        f"Epoch {epoch:02d}/{MAX_EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy:.4f} | "
        f"Val Acc: {val_metrics['accuracy']:.4f} | "
        f"Val Macro F1: {val_f1:.4f}"
    )

    if val_f1 > best_macro_f1:
        best_macro_f1 = val_f1
        best_epoch = epoch
        epochs_without_improvement = 0
        best_state = deepcopy(model.state_dict())

        torch.save({
            "model_state_dict": best_state,
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "architecture": "resnet18",
            "epoch": best_epoch,
            "validation_macro_f1": best_macro_f1,
            "normalization_mean": [0.485, 0.456, 0.406],
            "normalization_std": [0.229, 0.224, 0.225],
        }, CHECKPOINT)

        print("  -> Best checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("\nEarly stopping.")
        break


# ============================================================
# FINAL TEST
# ============================================================

if best_state is None:
    raise RuntimeError("Training did not produce a checkpoint.")

model.load_state_dict(best_state)

test_actual, test_predicted, test_probabilities, test_paths = (
    evaluate(model, test_loader)
)

overall_metrics = calculate_metrics(
    test_actual, test_predicted
)

print("\n========== OVERALL TEST RESULTS ==========")

for name, value in overall_metrics.items():
    print(f"{name}: {value}")

print("\nClassification report:")
print(
    classification_report(
        test_actual,
        test_predicted,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
)

# Per-source test metrics.
test_results = test_df.reset_index(drop=True).copy()
test_results["actual_index"] = test_actual
test_results["predicted_index"] = test_predicted
test_results["normal_probability"] = test_probabilities[:, 0]
test_results["defective_probability"] = test_probabilities[:, 1]
test_results["predicted_label"] = [
    CLASS_NAMES[i] for i in test_predicted
]

source_results = {}

print("\n========== TEST RESULTS BY SOURCE ==========")

for source in sorted(test_results["source"].unique()):
    subset = test_results[test_results["source"] == source]

    metrics = calculate_metrics(
        subset["actual_index"].to_numpy(),
        subset["predicted_index"].to_numpy(),
    )

    source_results[source] = metrics

    print(f"\n{source} ({len(subset)} images)")

    for name, value in metrics.items():
        print(f"{name}: {value}")

# Save reports.
pd.DataFrame(history).to_csv(
    REPORT_DIR / "training_history.csv",
    index=False,
)

test_results.to_csv(
    REPORT_DIR / "test_predictions.csv",
    index=False,
)

pd.DataFrame(
    confusion_matrix(test_actual, test_predicted, labels=[0, 1]),
    index=["actual_normal", "actual_defective"],
    columns=["predicted_normal", "predicted_defective"],
).to_csv(REPORT_DIR / "confusion_matrix.csv")

final_report = {
    "best_epoch": best_epoch,
    "best_validation_macro_f1": best_macro_f1,
    "overall_test_metrics": overall_metrics,
    "test_metrics_by_source": source_results,
    "training_images": len(train_df),
    "validation_images": len(val_df),
    "test_images": len(test_df),
    "checkpoint": str(CHECKPOINT),
    "warning": (
        "Research prototype only. Results on these datasets do not "
        "establish performance on unseen medicine types or production "
        "inspection conditions."
    ),
}

with open(REPORT_DIR / "test_metrics.json", "w", encoding="utf-8") as f:
    json.dump(final_report, f, indent=4)

print("\n========== TRAINING COMPLETE ==========")
print("Best epoch:", best_epoch)
print("Best validation macro F1:", round(best_macro_f1, 4))
print("Model:", CHECKPOINT)
print("Reports:", REPORT_DIR)
