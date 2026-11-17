from pathlib import Path
import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT
    / "runs"
    / "tablet_detection"
    / "tablet_yolov8n_pilot"
    / "weights"
    / "best.pt"
)

DATA_YAML = (
    ROOT
    / "datasets"
    / "tablet_detection"
    / "prepared"
    / "data.yaml"
)

OUTPUT_DIR = ROOT / "runs" / "tablet_detection"

# Read test image location from data.yaml
with open(DATA_YAML, "r", encoding="utf-8") as file:
    data = yaml.safe_load(file)

dataset_root = Path(data.get("path", DATA_YAML.parent))

if not dataset_root.is_absolute():
    dataset_root = (DATA_YAML.parent / dataset_root).resolve()

test_source = data["test"]

if isinstance(test_source, str):
    test_source = Path(test_source)
    if not test_source.is_absolute():
        test_source = dataset_root / test_source
else:
    test_source = [
        str((dataset_root / item).resolve())
        if not Path(item).is_absolute()
        else item
        for item in test_source
    ]

print("=" * 60)
print("PHARMASIGHT AI - TEST PREDICTION VISUAL INSPECTION")
print("=" * 60)

print("Loading trained detector...")
model = YOLO(str(MODEL_PATH))

print("Running predictions on reserved test images...")

results = model.predict(
    source=str(test_source) if isinstance(test_source, Path) else test_source,
    imgsz=640,
    conf=0.25,
    device="cpu",
    save=True,
    show=False,
    project=str(OUTPUT_DIR),
    name="tablet_detector_test_predictions",
    exist_ok=True,
)

print("\n========== PREDICTION SUMMARY ==========")

for result in results:
    filename = Path(result.path).name
    count = len(result.boxes) if result.boxes is not None else 0
    print(f"{filename}: {count} tablet(s) detected")

print("\n========== COMPLETE ==========")
print("Images saved in:")
print(OUTPUT_DIR / "tablet_detector_test_predictions")