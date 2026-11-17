from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "runs" / "tablet_detection"
    / "tablet_yolov8n_pilot" / "weights" / "best.pt"
)

DATA_YAML = (
    ROOT / "datasets" / "tablet_detection"
    / "prepared_stratified" / "data.yaml"
)

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

print("=" * 60)
print("PHARMASIGHT AI - FINAL TABLET DETECTOR TEST")
print("=" * 60)

model = YOLO(str(MODEL_PATH))

print("\nEvaluating the reserved test dataset...")

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    plots=True,
    project=str(ROOT / "runs" / "tablet_detection"),
    name="tablet_detector_test",
    exist_ok=True
)

print("\n========== TEST RESULTS ==========")
print("Precision:", round(float(metrics.box.mp), 4))
print("Recall:", round(float(metrics.box.mr), 4))
print("mAP50:", round(float(metrics.box.map50), 4))
print("mAP50-95:", round(float(metrics.box.map), 4))

print("\n========== SAVED MODEL ==========")
print(MODEL_PATH)

print("\n========== TEST REPORT ==========")
print(ROOT / "runs" / "tablet_detection" / "tablet_detector_test")

