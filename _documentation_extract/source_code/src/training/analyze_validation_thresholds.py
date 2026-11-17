from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch import nn
from PIL import Image
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "datasets" / "dataset_manifest.csv"
CHECKPOINT = ROOT / "models" / "tablet_classifier" / "best_model.pth"
OUTPUT = ROOT / "reports" / "training"

class ValidationDataset(Dataset):
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
        path = ROOT / str(row["image_path"])
        image = Image.open(path).convert("RGB")
        image = self.transform(image)
        label = 1 if row["label"] == "defective" else 0
        return image, label, str(row["image_path"])

def main():
    if not CHECKPOINT.is_file():
        raise SystemExit(f"Model checkpoint not found: {CHECKPOINT}")

    df = pd.read_csv(MANIFEST).fillna("")
    df = df[
        (df["task"].astype(str).str.strip().str.lower() == "tablet") &
        (df["split"].astype(str).str.strip().str.lower() == "validation")
    ].copy()

    if len(df) != 60:
        raise SystemExit(f"Expected 60 validation images, found {len(df)}")

    if not set(df["label"]).issubset({"normal", "defective"}):
        raise SystemExit("Unexpected validation labels.")

    device = torch.device("cpu")
    checkpoint = torch.load(
        CHECKPOINT, map_location=device, weights_only=False
    )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    loader = DataLoader(
        ValidationDataset(df),
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    all_labels = []
    all_probabilities = []
    all_paths = []

    with torch.no_grad():
        for images, labels, paths in loader:
            logits = model(images.to(device))
            probabilities = torch.softmax(logits, dim=1)[:, 1]

            all_labels.extend(labels.numpy().tolist())
            all_probabilities.extend(probabilities.numpy().tolist())
            all_paths.extend(paths)

    y_true = np.array(all_labels)
    y_prob = np.array(all_probabilities)

    predictions = pd.DataFrame({
        "image_path": all_paths,
        "actual_label": [
            "defective" if x == 1 else "normal" for x in y_true
        ],
        "defective_probability": y_prob
    })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(
        OUTPUT / "validation_predictions.csv", index=False
    )

    results = []

    for threshold in np.round(np.arange(0.20, 0.801, 0.05), 2):
        y_pred = (y_prob >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).ravel()

        results.append({
            "threshold": threshold,
            "accuracy": accuracy_score(y_true, y_pred),
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
            "TN": tn,
            "FP": fp,
            "FN": fn,
            "TP": tp
        })

    report = pd.DataFrame(results)
    report.to_csv(OUTPUT / "validation_threshold_analysis.csv", index=False)

    print("\n========== VALIDATION THRESHOLD ANALYSIS ==========")
    print(report.round(4).to_string(index=False))
    print("\nSaved:")
    print(OUTPUT / "validation_predictions.csv")
    print(OUTPUT / "validation_threshold_analysis.csv")
    print("\nOnly validation images were evaluated.")
    print("No test images were loaded.")

if __name__ == "__main__":
    main()
