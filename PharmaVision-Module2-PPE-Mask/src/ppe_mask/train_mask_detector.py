"""Train / fine-tune a YOLOv8n mask detector (CPU friendly, same settings style as your tablet detector).

First training:
  python src/ppe_mask/train_mask_detector.py --data datasets/ppe_mask/prepared/data.yaml
Second stage (after adding your own webcam sessions to TRAIN via prepare_mask_dataset.py):
  python src/ppe_mask/train_mask_detector.py --data datasets/ppe_mask/prepared/data.yaml \
      --weights runs/ppe_mask/mask_yolov8n/weights/best.pt --name mask_yolov8n_stage2 --epochs 40
Best weights: runs/ppe_mask/<name>/weights/best.pt
"""
from __future__ import annotations
import argparse


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="datasets/ppe_mask/prepared/data.yaml")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--project", default="runs/ppe_mask")
    ap.add_argument("--name", default="mask_yolov8n")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    from ultralytics import YOLO
    model = YOLO(a.weights)
    model.train(data=a.data, epochs=a.epochs, patience=a.patience, imgsz=a.imgsz, batch=a.batch,
                device=a.device, workers=0, project=a.project, name=a.name, seed=a.seed,
                deterministic=True, amp=False, close_mosaic=10, plots=True)
    print(f"\nDone. Best weights: {a.project}/{a.name}/weights/best.pt")
    print("Do not tune on the test split. Evaluate once with evaluate_mask_detector.py and evaluate_gate.py.")


if __name__ == "__main__":
    main()
