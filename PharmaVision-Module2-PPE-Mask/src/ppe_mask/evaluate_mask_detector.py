"""Standard detection metrics (per-class precision/recall/AP) on a split of the prepared dataset.
  python src/ppe_mask/evaluate_mask_detector.py --weights runs/ppe_mask/mask_yolov8n/weights/best.pt --split test
Run on the TEST split once, after all choices are final.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ppe_mask/mask_yolov8n/weights/best.pt")
    ap.add_argument("--data", default="datasets/ppe_mask/prepared/data.yaml")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", type=Path, default=Path("reports/ppe_mask/detector_metrics.json"))
    a = ap.parse_args()
    from ultralytics import YOLO
    m = YOLO(a.weights).val(data=a.data, split=a.split, imgsz=a.imgsz, device=a.device, workers=0, plots=True)
    names = m.names
    per_class = {}
    for i, c in enumerate(m.box.ap_class_index):
        per_class[names[int(c)]] = {"precision": float(m.box.p[i]), "recall": float(m.box.r[i]),
                                    "ap50": float(m.box.ap50[i]), "ap50_95": float(m.box.ap[i])}
    result = {"split": a.split, "weights": a.weights, "map50": float(m.box.map50), "map50_95": float(m.box.map),
              "per_class": per_class,
              "caution": "Rare class mask_weared_incorrect has few examples; its numbers are unreliable. "
                         "Detection metrics are not the same as the operator-gate error rates; see evaluate_gate.py."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
