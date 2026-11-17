from pathlib import Path
from ultralytics import YOLO
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_YAML = (
    PROJECT_ROOT / "datasets" / "tablet_detection"
    / "prepared_stratified" / "data.yaml"
)

if not DATA_YAML.is_file():
    raise FileNotFoundError(f"Dataset configuration not found: {DATA_YAML}")

print("=" * 60)
print("PHARMASIGHT AI - TABLET DETECTOR TRAINING")
print("=" * 60)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Dataset:", DATA_YAML)

# Load a small pretrained object detector.
model = YOLO("yolov8n.pt")

# Fine-tune it for one class: tablet.
results = model.train(
    data=str(DATA_YAML),
    epochs=80,
    patience=20,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    pretrained=True,
    optimizer="AdamW",
    lr0=0.001,
    weight_decay=0.0005,
    seed=42,
    deterministic=True,
    amp=False,
    cache=False,
    close_mosaic=10,
    project=str(PROJECT_ROOT / "runs" / "tablet_detection"),
    name="tablet_yolov8n_pilot",
    exist_ok=True,
    save=True,
    plots=True,
    verbose=True
)

best_model = (
    PROJECT_ROOT / "runs" / "tablet_detection"
    / "tablet_yolov8n_pilot" / "weights" / "best.pt"
)

print("\n" + "=" * 60)
print("TRAINING FINISHED")
print("=" * 60)
print("Best model:", best_model)
print("Model exists:", best_model.is_file())


