import sys
from pathlib import Path

MODULE_ROOT = Path("PharmaVision-Module2-PPE-Mask").resolve()
sys.path.insert(0, str(MODULE_ROOT))
import json
import csv
from pathlib import Path

import cv2
from src.ppe_mask.gate import GateConfig
from src.ppe_mask.mask_module import MaskComplianceModule
from src.ppe_mask.evaluate_gate import evaluate

ROOT = Path(".")
WEIGHTS = ROOT / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt"
IMAGES = ROOT / "datasets/ppe_mask/prepared/images/val"
LABELS = ROOT / "datasets/ppe_mask/prepared/labels/val"
OUT = ROOT / "runs/ppe_mask/gate_validation_experiment"

OUT.mkdir(parents=True, exist_ok=True)

configs = {
    "baseline": GateConfig(
        ok_conf=0.60,
        bad_conf=0.35,
        min_face_area_frac=0.002,
        open_after_frames=8,
        close_after_frames=3,
    ),
    "sensitive": GateConfig(
        ok_conf=0.60,
        bad_conf=0.20,
        min_face_area_frac=0.001,
        open_after_frames=8,
        close_after_frames=3,
    ),
    "very_sensitive": GateConfig(
        ok_conf=0.60,
        bad_conf=0.15,
        min_face_area_frac=0.0005,
        open_after_frames=8,
        close_after_frames=3,
    ),
}

all_summaries = {}

for name, cfg in configs.items():
    print(f"\nRunning configuration: {name}")

    mod = MaskComplianceModule(
        weights=str(WEIGHTS),
        conf=0.10,
        imgsz=416,
        device="cpu",
        cfg=cfg,
    )

    def predict(path):
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Could not read image: {path}")
        h, w = image.shape[:2]
        return mod.detect(image), float(h * w)

    summary, rows = evaluate(IMAGES, LABELS, predict, cfg)

    all_summaries[name] = summary

    with open(OUT / f"{name}_rows.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "truth_compliant",
                "status",
                "predicted_compliant",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(summary, indent=2))

with open(OUT / "comparison.json", "w") as f:
    json.dump(all_summaries, f, indent=2)

print("\nExperiment complete.")
print("Results:", OUT.resolve())

