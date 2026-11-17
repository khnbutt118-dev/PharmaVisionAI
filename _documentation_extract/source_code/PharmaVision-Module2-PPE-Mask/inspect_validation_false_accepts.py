from pathlib import Path
import cv2

from src.ppe_mask.gate import (
    GateConfig, frame_status, CLASS_NAMES
)
from src.ppe_mask.mask_module import MaskComplianceModule

ROOT = Path.cwd()

images_dir = ROOT / "datasets/ppe_mask/prepared/images/val"
labels_dir = ROOT / "datasets/ppe_mask/prepared/labels/val"
weights = ROOT / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt"

files = [
    "maksssksksss113.png",
    "maksssksksss280.png",
    "maksssksksss619.png",
    "maksssksksss75.png",
]

cfg = GateConfig(
    ok_conf=0.60,
    bad_conf=0.15,
    min_face_area_frac=0.0005,
    open_after_frames=8,
    close_after_frames=3,
)

mod = MaskComplianceModule(
    weights=str(weights),
    conf=0.10,
    imgsz=416,
    device="cpu",
    cfg=cfg,
)

for filename in files:
    image_path = images_dir / filename
    label_path = labels_dir / (Path(filename).stem + ".txt")

    image = cv2.imread(str(image_path))
    if image is None:
        print("\nMISSING IMAGE:", filename)
        continue

    h, w = image.shape[:2]
    frame_area = float(h * w)

    print("\n" + "=" * 65)
    print("IMAGE:", filename)

    print("GROUND TRUTH:")
    if label_path.exists():
        for line in label_path.read_text().splitlines():
            if line.strip():
                class_id = int(line.split()[0])
                print(" ", CLASS_NAMES[class_id])
    else:
        print(" LABEL FILE MISSING")

    detections = mod.detect(image)

    print("\nMODEL DETECTIONS:")
    if not detections:
        print(" NO DETECTIONS")

    for d in detections:
        x1, y1, x2, y2 = d.box
        area_fraction = max(0, x2-x1) * max(0, y2-y1) / frame_area
        retained = area_fraction >= cfg.min_face_area_frac

        print(
            f" {d.cls:24s} "
            f"confidence={d.conf:.3f} "
            f"area_fraction={area_fraction:.6f} "
            f"retained={retained}"
        )

    print("\nGATE STATUS:", frame_status(
        detections, cfg, frame_area=frame_area
    ))

print("\nDiagnostic completed.")
