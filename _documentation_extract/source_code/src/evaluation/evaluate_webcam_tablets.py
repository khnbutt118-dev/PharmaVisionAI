import csv
import re
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = ROOT / "datasets" / "tablet_detection" / "images" / "all"
DETECTOR_PATH = ROOT / "runs" / "tablet_detection" / "tablet_yolov8n_pilot" / "weights" / "best.pt"
CLASSIFIER_PATH = ROOT / "models" / "combined_binary" / "best_model.pth"

REPORT_DIR = ROOT / "reports"
CSV_PATH = REPORT_DIR / "tablet_webcam_baseline.csv"
MATRIX_PATH = REPORT_DIR / "tablet_webcam_confusion_matrix.png"
TEXT_PATH = REPORT_DIR / "tablet_webcam_evaluation.txt"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
YOLO_CONFIDENCE = 0.25
BOX_PADDING = 0.20
REVIEW_LOW = 0.40
REVIEW_HIGH = 0.60

TRANSFORM = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

def ground_truth(number):
    # Dataset manifest:
    # 0001-0060 normal; 0061-0080 chipped; 0081-0101 normal
    return "defective" if 61 <= number <= 80 else "normal"

def expand_box(x1, y1, x2, y2, frame_width, frame_height, padding):
    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    pad_x = int(round(width * padding))
    pad_y = int(round(height * padding))
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_width - 1, x2 + pad_x),
        min(frame_height - 1, y2 + pad_y)
    )

def load_classifier():
    checkpoint = torch.load(
        CLASSIFIER_PATH, map_location="cpu", weights_only=False
    )
    classifier = models.resnet18(weights=None)
    classifier.fc = nn.Linear(classifier.fc.in_features, 2)
    classifier.load_state_dict(checkpoint["model_state_dict"], strict=True)
    classifier.to(DEVICE).eval()

    names = [x.strip().lower() for x in checkpoint["class_names"]]
    defective_index = names.index("defective")
    return classifier, defective_index

def classify_crop(crop, classifier, defective_index):
    if crop.size == 0:
        return None

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    tensor = TRANSFORM(Image.fromarray(rgb)).unsqueeze(0).to(DEVICE)

    with torch.inference_mode():
        probabilities = torch.softmax(classifier(tensor), dim=1)[0]

    return float(probabilities[defective_index].item())

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    classifier, defective_index = load_classifier()
    detector = YOLO(str(DETECTOR_PATH))

    images = sorted(
        IMAGE_DIR.glob("*"),
        key=lambda p: int(re.search(r"(\d+)", p.stem).group(1))
        if re.search(r"(\d+)", p.stem) else 999999
    )
    images = [p for p in images if p.suffix.lower() in
              {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]

    rows = []
    print(f"Device: {DEVICE}")
    print(f"Images: {len(images)}")
    print("Evaluating webcam dataset...")

    for i, path in enumerate(images, 1):
        match = re.search(r"(\d+)", path.stem)
        number = int(match.group(1)) if match else i
        truth = ground_truth(number)

        frame = cv2.imread(str(path))
        if frame is None:
            print(f"SKIP unreadable: {path.name}")
            continue

        h, w = frame.shape[:2]
        result = detector.predict(
            source=frame,
            conf=YOLO_CONFIDENCE,
            verbose=False
        )[0]

        p_defective = None
        detection_count = len(result.boxes)

        if detection_count:
            # Use highest-confidence detector box
            best_idx = int(result.boxes.conf.argmax().item())
            x1, y1, x2, y2 = map(
                int, result.boxes.xyxy[best_idx].tolist()
            )
            x1, y1, x2, y2 = expand_box(
                x1, y1, x2, y2, w, h, BOX_PADDING
            )
            crop = frame[y1:y2, x1:x2]
            p_defective = classify_crop(crop, classifier, defective_index)

        if p_defective is None:
            prediction = "review"
            confidence = None
        elif REVIEW_LOW <= p_defective <= REVIEW_HIGH:
            prediction = "review"
            confidence = max(p_defective, 1 - p_defective)
        elif p_defective > REVIEW_HIGH:
            prediction = "defective"
            confidence = p_defective
        else:
            prediction = "normal"
            confidence = 1 - p_defective

        rows.append({
            "filename": path.name,
            "ground_truth": truth,
            "prediction": prediction,
            "p_defective": p_defective,
            "confidence": confidence,
            "detections": detection_count,
            "status": "OK" if detection_count else "NO_DETECTION"
        })

        print(
            f"[{i:03}/{len(images):03}] {path.name:20} "
            f"GT={truth:9} Pred={prediction:9} "
            f"P(def)={p_defective if p_defective is not None else 'N/A'}"
        )

    fields = [
        "filename", "ground_truth", "prediction",
        "p_defective", "confidence", "detections", "status"
    ]
    with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    # Evaluate only decisive normal/defective predictions.
    decisive = [r for r in rows if r["prediction"] in ("normal", "defective")]
    y_true = [r["ground_truth"] for r in decisive]
    y_pred = [r["prediction"] for r in decisive]

    labels = ["normal", "defective"]
    matrix = np.zeros((2, 2), dtype=int)
    for truth, pred in zip(y_true, y_pred):
        matrix[labels.index(truth), labels.index(pred)] += 1

    tn, fp = matrix[0]
    fn, tp = matrix[1]

    total = len(rows)
    covered = len(decisive)
    correct = sum(a == b for a, b in zip(y_true, y_pred))
    accuracy = correct / covered if covered else 0
    coverage = covered / total if total else 0
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    specificity = tn / (tn + fp) if tn + fp else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0

    plt.figure(figsize=(6, 5))
    plt.imshow(matrix)
    plt.title("Webcam Tablet Binary Confusion Matrix")
    plt.xticks([0, 1], labels)
    plt.yticks([0, 1], labels)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    for r in range(2):
        for c in range(2):
            plt.text(c, r, str(matrix[r, c]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(MATRIX_PATH, dpi=160)
    plt.close()

    report = f"""
PHARMASIGHT AI - WEBCAM TABLET BASELINE
========================================
Images evaluated: {total}
Decisive predictions: {covered}
Review/no-detection cases: {total - covered}
Coverage: {coverage:.4f}
Accuracy among decisive predictions: {accuracy:.4f}

Confusion matrix (decisive predictions):
TN={tn}  FP={fp}
FN={fn}  TP={tp}

Defective precision: {precision:.4f}
Defective recall: {recall:.4f}
Normal specificity: {specificity:.4f}
Defective F1: {f1:.4f}

Thresholds:
Normal: P(defective) < {REVIEW_LOW}
Review: {REVIEW_LOW} <= P(defective) <= {REVIEW_HIGH}
Defective: P(defective) > {REVIEW_HIGH}

NOTE:
This is a small webcam dataset evaluation, not an independent
clinical validation. Chipped examples are from one tablet type.
""".strip()

    TEXT_PATH.write_text(report, encoding="utf-8")
    print("\n" + report)
    print("\nGenerated:")
    print(CSV_PATH)
    print(MATRIX_PATH)
    print(TEXT_PATH)

if __name__ == "__main__":
    main()
