from pathlib import Path
import cv2
from ultralytics import YOLO

ROOT = Path(".")
IMAGE_DIR = ROOT / "datasets/ppe_mask/prepared/images/test"
WEIGHTS = ROOT / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt"
CSV_FILE = ROOT / "runs/ppe_mask/gate_evaluation/gate_rows.csv"
OUT = ROOT / "runs/ppe_mask/false_accept_review"

OUT.mkdir(parents=True, exist_ok=True)

false_accepts = [
    "maksssksksss126.png",
    "maksssksksss349.png",
    "maksssksksss384.png",
    "maksssksksss486.png",
    "maksssksksss535.png",
    "maksssksksss611.png",
    "maksssksksss715.png",
    "maksssksksss823.png",
]

model = YOLO(str(WEIGHTS))

for name in false_accepts:
    image_path = IMAGE_DIR / name

    if not image_path.exists():
        print("MISSING:", image_path)
        continue

    result = model.predict(
        source=str(image_path),
        imgsz=416,
        device="cpu",
        conf=0.10,
        verbose=False
    )[0]

    annotated = result.plot()

    cv2.imwrite(str(OUT / name), annotated)
    print("Saved:", OUT / name)

print("\nReview images created:", OUT.resolve())
