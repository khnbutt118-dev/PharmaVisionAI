from pathlib import Path
from ultralytics import YOLO

images = [
    "maksssksksss126.png",
    "maksssksksss349.png",
    "maksssksksss384.png",
    "maksssksksss486.png",
    "maksssksksss535.png",
    "maksssksksss611.png",
    "maksssksksss715.png",
    "maksssksksss823.png",
]

model = YOLO("runs/ppe_mask/mask_yolov8n_full/weights/best.pt")
names = model.names

for image in images:
    path = Path("datasets/ppe_mask/prepared/images/test") / image
    result = model.predict(str(path), imgsz=416, device="cpu",
                           conf=0.10, verbose=False)[0]

    print("\nIMAGE:", image)
    if result.boxes is None or len(result.boxes) == 0:
        print("NO DETECTIONS")
        continue

    for box in result.boxes:
        cls = int(box.cls.item())
        conf = float(box.conf.item())
        xyxy = box.xyxy[0].tolist()
        print(f"{names[cls]} | confidence={conf:.3f} | box={xyxy}")
