from pathlib import Path
from PIL import Image
from src.ppe_mask.mask_module import MaskComplianceModule
from src.ppe_mask.gate import GateConfig, frame_status

root = Path.cwd()
weights = root / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt"
image_dir = root / "datasets/ppe_mask/prepared/images/val"

names = [
    "maksssksksss113.png",
    "maksssksksss75.png",
    "maksssksksss280.png",
    "maksssksksss619.png",
]

cfg = GateConfig(
    ok_conf=0.60,
    bad_conf=0.15,
    min_face_area_frac=0.0005
)

for imgsz in [416, 640, 832]:
    print("\n" + "=" * 65)
    print("INFERENCE SIZE:", imgsz)
    print("=" * 65)

    module = MaskComplianceModule(
        weights=str(weights),
        conf=0.10,
        imgsz=imgsz,
        device="cpu",
        cfg=cfg
    )

    for name in names:
        path = image_dir / name

        with Image.open(path) as im:
            frame = im.convert("RGB")

        import numpy as np
        frame = np.array(frame)[:, :, ::-1].copy()

        detections = module.detect(frame)

        status = frame_status(
            detections,
            cfg,
            frame_area=frame.shape[0] * frame.shape[1]
        )

        print(f"\n{name}")
        print("Status:", status)
        print("Detections:", len(detections))

        for d in detections:
            print(
                f"  {d.cls:25} "
                f"confidence={d.conf:.3f} "
                f"box={d.box}"
            )
