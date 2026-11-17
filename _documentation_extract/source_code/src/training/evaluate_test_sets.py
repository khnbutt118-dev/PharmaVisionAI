from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score
)

ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT = ROOT / "models" / "tablet_classifier" / "best_model.pth"
EVALUATION_DIR = ROOT / "reports" / "dataset_audit" / "evaluation"
OUTPUT_DIR = ROOT / "reports" / "evaluation"

THRESHOLD = 0.25

class TestDataset(Dataset):
    def __init__(self, frame):
        self.frame = frame.reset_index(drop=True)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image_path = ROOT / str(row["image_path"])

        with Image.open(image_path) as img:
            image = self.transform(img.convert("RGB"))

        label = 1 if str(row["label"]).strip().lower() == "defective" else 0
        return image, label, str(row["image_path"])

def evaluate_protocol(model, csv_path, protocol_name, device):
    if not csv_path.is_file():
        raise SystemExit(f"Evaluation file not found: {csv_path}")

    frame = pd.read_csv(csv_path).fillna("")

    required = {"image_path", "label"}
    if not required.issubset(frame.columns):
        raise SystemExit(f"{csv_path.name} must contain {sorted(required)}")

    if frame.empty:
        raise SystemExit(f"{csv_path.name} is empty.")

    if not set(frame["label"].str.strip().str.lower()).issubset(
        {"normal", "defective"}
    ):
        raise SystemExit(f"Unexpected labels in {csv_path.name}")

    missing = [
        str(ROOT / p) for p in frame["image_path"]
        if not (ROOT / p).is_file()
    ]
    if missing:
        raise SystemExit(
            f"Missing test image: {missing[0]}"
        )

    loader = DataLoader(
        TestDataset(frame),
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    y_true = []
    probabilities = []
    paths = []

    model.eval()
    with torch.no_grad():
        for images, labels, image_paths in loader:
            logits = model(images.to(device))
            probs = torch.softmax(logits, dim=1)[:, 1]

            y_true.extend(labels.numpy().tolist())
            probabilities.extend(probs.cpu().numpy().tolist())
            paths.extend(image_paths)

    y_true = np.array(y_true)
    probabilities = np.array(probabilities)
    y_pred = (probabilities >= THRESHOLD).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    ).ravel()

    metrics = {
        "protocol": protocol_name,
        "images": len(y_true),
        "threshold": THRESHOLD,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "precision_defective": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "recall_defective": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "specificity_normal": tn / (tn + fp) if tn + fp else 0,
        "false_pass_rate": fn / (fn + tp) if fn + tp else 0,
        "f1_defective": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    }

    predictions = pd.DataFrame({
        "image_path": paths,
        "actual_label": [
            "defective" if x == 1 else "normal" for x in y_true
        ],
        "defective_probability": probabilities,
        "predicted_label": [
            "defective" if x == 1 else "normal" for x in y_pred
        ],
        "threshold": THRESHOLD
    })

    safe_name = protocol_name.lower().replace(" ", "_")
    predictions.to_csv(
        OUTPUT_DIR / f"{safe_name}_predictions.csv",
        index=False
    )

    return metrics

def main():
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Checkpoint not found: {CHECKPOINT}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cpu")
    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    print("Device:", device)
    print("Checkpoint:", CHECKPOINT)
    print("Fixed threshold:", THRESHOLD)
    print("No model training or threshold tuning will occur.")

    results = []

    protocols = [
        (
            "Full Test Set",
            EVALUATION_DIR / "test_all_images.csv"
        ),
        (
            "Sample-Aware Test Set",
            EVALUATION_DIR / "test_one_per_confirmed_sample.csv"
        )
    ]

    for name, csv_path in protocols:
        print(f"\nEvaluating: {name}")
        result = evaluate_protocol(
            model, csv_path, name, device
        )
        results.append(result)

        print(pd.Series(result).to_string())

    report = pd.DataFrame(results)
    report.to_csv(
        OUTPUT_DIR / "final_test_metrics.csv",
        index=False
    )

    with open(
        OUTPUT_DIR / "evaluation_protocol.json",
        "w",
        encoding="utf-8"
    ) as f:
        json.dump({
            "architecture": "resnet18",
            "checkpoint": str(CHECKPOINT),
            "threshold": THRESHOLD,
            "threshold_selected_on": "validation set only",
            "test_used_for_model_selection": False,
            "protocols": [
                "test_all_images.csv",
                "test_one_per_confirmed_sample.csv"
            ]
        }, f, indent=2)

    print("\n========== FINAL TEST SUMMARY ==========")
    print(report.round(4).to_string(index=False))
    print("\nReports saved in:", OUTPUT_DIR)
    print("Test evaluation complete.")
    print("Do not use these results to tune this model.")

if __name__ == "__main__":
    main()
