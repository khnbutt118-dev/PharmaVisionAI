from pathlib import Path
from collections import Counter
import csv
import json
import cv2

from src.ppe_mask.mask_module import MaskComplianceModule
from src.ppe_mask.gate import GateConfig, frame_status

ROOT = Path.cwd()

WEIGHTS = ROOT / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt"
IMAGES = ROOT / "datasets/ppe_mask/prepared/images/val"
LABELS = ROOT / "datasets/ppe_mask/prepared/labels/val"
OUT = ROOT / "runs/ppe_mask/resolution_validation_experiment"

OUT.mkdir(parents=True, exist_ok=True)

CONFIG = GateConfig(
    ok_conf=0.60,
    bad_conf=0.15,
    min_face_area_frac=0.0005
)

IMAGE_FILES = sorted(IMAGES.glob("*.png"))

if not IMAGE_FILES:
    raise RuntimeError(f"No PNG validation images found: {IMAGES}")

all_summaries = {}

for imgsz in [416, 640, 832]:

    print("\n" + "=" * 65)
    print(f"VALIDATING AT {imgsz} x {imgsz}")
    print("=" * 65)

    module = MaskComplianceModule(
        weights=str(WEIGHTS),
        conf=0.10,
        imgsz=imgsz,
        device="cpu",
        cfg=CONFIG
    )

    rows = []
    statuses = Counter()

    truth_compliant_count = 0
    truth_not_compliant_count = 0
    false_accepts = 0
    false_rejects = 0

    for index, image_path in enumerate(IMAGE_FILES, 1):

        label_path = LABELS / f"{image_path.stem}.txt"

        if not label_path.exists():
            raise FileNotFoundError(label_path)

        label_lines = [
            line.strip()
            for line in label_path.read_text().splitlines()
            if line.strip()
        ]

        gt_classes = [
            int(line.split()[0])
            for line in label_lines
        ]

        truth_compliant = (
            len(gt_classes) > 0
            and all(cls == 0 for cls in gt_classes)
        )

        frame = cv2.imread(str(image_path))

        if frame is None:
            raise RuntimeError(f"Cannot read image: {image_path}")

        detections = module.detect(frame)

        status = frame_status(
            detections,
            CONFIG,
            frame_area=frame.shape[0] * frame.shape[1]
        )

        predicted_compliant = status == "COMPLIANT"

        if truth_compliant:
            truth_compliant_count += 1
        else:
            truth_not_compliant_count += 1

        if not truth_compliant and predicted_compliant:
            false_accepts += 1

        if truth_compliant and not predicted_compliant:
            false_rejects += 1

        statuses[status] += 1

        rows.append({
            "image": image_path.name,
            "truth_compliant": truth_compliant,
            "predicted_compliant": predicted_compliant,
            "status": status,
            "detections": len(detections)
        })

        if index % 20 == 0 or index == len(IMAGE_FILES):
            print(f"Processed {index}/{len(IMAGE_FILES)}")

    far = (
        false_accepts / truth_not_compliant_count * 100
        if truth_not_compliant_count else 0
    )

    frr = (
        false_rejects / truth_compliant_count * 100
        if truth_compliant_count else 0
    )

    summary = {
        "imgsz": imgsz,
        "images": len(rows),
        "truth_compliant": truth_compliant_count,
        "truth_not_compliant": truth_not_compliant_count,
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "false_accept_rate_percent": round(far, 2),
        "false_reject_rate_percent": round(frr, 2),
        "statuses": dict(statuses)
    }

    all_summaries[str(imgsz)] = summary

    csv_path = OUT / f"resolution_{imgsz}_rows.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print("\nRESULT:")
    print(json.dumps(summary, indent=2))

with (OUT / "comparison.json").open("w", encoding="utf-8") as f:
    json.dump(all_summaries, f, indent=2)

print("\n" + "=" * 65)
print("FINAL RESOLUTION COMPARISON")
print("=" * 65)

for size, result in all_summaries.items():
    print(
        f"{size}: "
        f"False accepts={result['false_accepts']}, "
        f"FAR={result['false_accept_rate_percent']}%, "
        f"False rejects={result['false_rejects']}, "
        f"FRR={result['false_reject_rate_percent']}%"
    )

print("\nSaved results to:", OUT)
