from pathlib import Path
import json
import random

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42
EPOCHS = 25
BATCH_SIZE = 8
PATIENCE = 5
LEARNING_RATE = 0.0001
IMAGE_SIZE = 224

ROOT = Path(".")
MANIFEST = ROOT / "datasets/external/pillQC_manifest.csv"

OUT = ROOT / "reports/training/pillqc_multiclass"
MODEL_DIR = ROOT / "models/pillqc_multiclass"

OUT.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

CLASS_NAMES = ["normal", "contamination", "chipping"]
CLASS_TO_ID = {name: i for i, name in enumerate(CLASS_NAMES)}

# --------------------------------------------------
# REPRODUCIBILITY
# --------------------------------------------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

# --------------------------------------------------
# LOAD AND VALIDATE MANIFEST
# --------------------------------------------------

if not MANIFEST.exists():
    raise FileNotFoundError(f"Manifest not found: {MANIFEST}")

df = pd.read_csv(MANIFEST)

required = {"image_path", "task", "label", "defect_type"}
missing = required - set(df.columns)

if missing:
    raise ValueError(f"Manifest is missing columns: {sorted(missing)}")

df = df[df["task"] == "tablet"].copy()

def map_class(row):
    defect = str(row["defect_type"]).strip().lower()
    label = str(row["label"]).strip().lower()

    if label == "normal" or defect == "none":
        return "normal"
    if defect == "contamination":
        return "contamination"
    if defect == "chipping":
        return "chipping"
    return None

df["class_name"] = df.apply(map_class, axis=1)
df = df[df["class_name"].isin(CLASS_NAMES)].copy()

if df.empty:
    raise ValueError("No usable tablet records found.")

df["image_path"] = df["image_path"].astype(str)

missing_files = [
    path for path in df["image_path"]
    if not Path(path).is_file()
]

if missing_files:
    raise FileNotFoundError(
        f"{len(missing_files)} image files are missing. "
        f"First missing path: {missing_files[0]}"
    )

print("\n========== DATASET ==========")
print("Usable images:", len(df))
print(df["class_name"].value_counts().reindex(CLASS_NAMES))

if set(df["class_name"].unique()) != set(CLASS_NAMES):
    raise ValueError("All three classes must be present.")

# --------------------------------------------------
# STRATIFIED TRAIN / VALIDATION / TEST SPLIT
# --------------------------------------------------

train_val, test_df = train_test_split(
    df,
    test_size=0.15,
    random_state=SEED,
    stratify=df["class_name"]
)

train_df, val_df = train_test_split(
    train_val,
    test_size=0.1785714286,
    random_state=SEED,
    stratify=train_val["class_name"]
)

train_df = train_df.copy()
val_df = val_df.copy()
test_df = test_df.copy()

train_df["split"] = "train"
val_df["split"] = "validation"
test_df["split"] = "test"

split_df = pd.concat(
    [train_df, val_df, test_df],
    ignore_index=True
)

split_path = OUT / "pillqc_multiclass_split_manifest.csv"
split_df.to_csv(split_path, index=False)

print("\n========== SPLIT COUNTS ==========")
for name, part in [
    ("TRAIN", train_df),
    ("VALIDATION", val_df),
    ("TEST", test_df)
]:
    print(f"\n{name}: {len(part)}")
    print(part["class_name"].value_counts().reindex(CLASS_NAMES))

# --------------------------------------------------
# IMAGE TRANSFORMS
# --------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(8),
    transforms.ColorJitter(
        brightness=0.08,
        contrast=0.08,
        saturation=0.08
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

class PillDataset(Dataset):
    def __init__(self, frame, transform):
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        path = Path(row["image_path"])

        with Image.open(path) as image:
            image = image.convert("RGB")

        image = self.transform(image)
        label = CLASS_TO_ID[row["class_name"]]

        return image, label, str(path)


train_loader = DataLoader(
    PillDataset(train_df, train_transform),
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    PillDataset(val_df, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    PillDataset(test_df, eval_transform),
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

# --------------------------------------------------
# MODEL
# --------------------------------------------------

weights = models.ResNet18_Weights.DEFAULT
model = models.resnet18(weights=weights)

model.fc = nn.Linear(
    model.fc.in_features,
    len(CLASS_NAMES)
)

model = model.to(device)

# Give the less common classes more influence.
class_counts = (
    train_df["class_name"]
    .value_counts()
    .reindex(CLASS_NAMES)
    .values
)

class_weights = len(train_df) / (
    len(CLASS_NAMES) * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32,
    device=device
)

criterion = nn.CrossEntropyLoss(weight=class_weights)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=0.0001
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=2
)

# --------------------------------------------------
# EVALUATION FUNCTION
# --------------------------------------------------

def evaluate(loader):
    model.eval()

    actual = []
    predicted = []
    probabilities = []
    paths = []

    with torch.no_grad():
        for images, labels, batch_paths in loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            actual.extend(labels.numpy().tolist())
            predicted.extend(preds.cpu().numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            paths.extend(batch_paths)

    macro_f1 = f1_score(
        actual,
        predicted,
        average="macro",
        zero_division=0
    )

    return actual, predicted, probabilities, paths, macro_f1


# --------------------------------------------------
# TRAINING
# --------------------------------------------------

best_val_f1 = -1.0
epochs_without_improvement = 0
history = []

checkpoint_path = MODEL_DIR / "best_model.pth"

print("\n========== TRAINING ==========")

for epoch in range(1, EPOCHS + 1):
    model.train()

    running_loss = 0.0
    total = 0

    for images, labels, _ in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

    train_loss = running_loss / max(total, 1)

    _, _, _, _, val_f1 = evaluate(val_loader)
    scheduler.step(val_f1)

    current_lr = optimizer.param_groups[0]["lr"]

    history.append({
        "epoch": epoch,
        "train_loss": train_loss,
        "validation_macro_f1": val_f1,
        "learning_rate": current_lr
    })

    print(
        f"Epoch {epoch:02d}/{EPOCHS} | "
        f"Loss: {train_loss:.4f} | "
        f"Validation macro F1: {val_f1:.4f} | "
        f"LR: {current_lr:.7f}"
    )

    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        epochs_without_improvement = 0

        torch.save({
            "model_state_dict": model.state_dict(),
            "class_names": CLASS_NAMES,
            "image_size": IMAGE_SIZE,
            "best_validation_macro_f1": best_val_f1
        }, checkpoint_path)

        print("  Best model checkpoint saved.")

    else:
        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:
        print("Early stopping.")
        break

pd.DataFrame(history).to_csv(
    OUT / "training_history.csv",
    index=False
)

# --------------------------------------------------
# FINAL TEST - RUN ONLY AFTER TRAINING
# --------------------------------------------------

print("\n========== FINAL TEST ==========")

checkpoint = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])

actual, predicted, probabilities, paths, test_macro_f1 = evaluate(
    test_loader
)

accuracy = accuracy_score(actual, predicted)
balanced_accuracy = balanced_accuracy_score(actual, predicted)

report = classification_report(
    actual,
    predicted,
    labels=list(range(len(CLASS_NAMES))),
    target_names=CLASS_NAMES,
    output_dict=True,
    zero_division=0
)

matrix = confusion_matrix(
    actual,
    predicted,
    labels=list(range(len(CLASS_NAMES)))
)

metrics = {
    "test_accuracy": float(accuracy),
    "test_balanced_accuracy": float(balanced_accuracy),
    "test_macro_f1": float(test_macro_f1),
    "class_names": CLASS_NAMES,
    "classification_report": report,
    "confusion_matrix": matrix.tolist(),
    "test_images": len(actual),
    "best_validation_macro_f1": float(best_val_f1)
}

with (OUT / "test_metrics.json").open("w", encoding="utf-8") as file:
    json.dump(metrics, file, indent=2)

pd.DataFrame(
    matrix,
    index=[f"actual_{name}" for name in CLASS_NAMES],
    columns=[f"predicted_{name}" for name in CLASS_NAMES]
).to_csv(OUT / "confusion_matrix.csv")

prediction_df = pd.DataFrame({
    "image_path": paths,
    "actual_class": [CLASS_NAMES[i] for i in actual],
    "predicted_class": [CLASS_NAMES[i] for i in predicted],
    "normal_probability": [p[0] for p in probabilities],
    "contamination_probability": [p[1] for p in probabilities],
    "chipping_probability": [p[2] for p in probabilities]
})

prediction_df.to_csv(
    OUT / "test_predictions.csv",
    index=False
)

print("Test accuracy:", round(accuracy, 4))
print("Test balanced accuracy:", round(balanced_accuracy, 4))
print("Test macro F1:", round(test_macro_f1, 4))

print("\n========== CLASS REPORT ==========")
for name in CLASS_NAMES:
    item = report[name]
    print(
        f"{name}: "
        f"precision={item['precision']:.3f}, "
        f"recall={item['recall']:.3f}, "
        f"F1={item['f1-score']:.3f}, "
        f"support={item['support']}"
    )

print("\n========== OUTPUT FILES ==========")
print("Split manifest:", split_path)
print("Best model:", checkpoint_path)
print("Metrics:", OUT / "test_metrics.json")
print("Confusion matrix:", OUT / "confusion_matrix.csv")
print("Predictions:", OUT / "test_predictions.csv")
print("Training history:", OUT / "training_history.csv")
print("\nTraining and evaluation completed.")
