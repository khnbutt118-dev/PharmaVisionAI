"""Stand-alone live demo: webcam -> mask detector -> NO MASK box + gate status.
Run from the project root:  python src/ppe_mask/live_mask_demo.py
Keys: Q / Esc = quit, S = save snapshot to reports/ppe_mask/
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
from src.ppe_mask.mask_module import MaskComplianceModule  # noqa: E402


def open_camera(index: int):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(index)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {index}. Try --camera 1.")
    return cap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="runs/ppe_mask/mask_yolov8n/weights/best.pt")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    a = ap.parse_args()
    if not Path(a.weights).exists():
        raise SystemExit(f"Weights not found: {a.weights}. Train first (see docs/PPE_MASK_MODULE.md).")
    mod = MaskComplianceModule(a.weights, conf=a.conf, imgsz=a.imgsz)
    cap = open_camera(a.camera)
    out_dir = Path("reports/ppe_mask")
    out_dir.mkdir(parents=True, exist_ok=True)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)  # mirror view; boxes/text are drawn after the flip so they read correctly
        annotated, info = mod.process(frame)
        cv2.imshow("PharmaVision AI - mask check", annotated)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord("s"):
            p = out_dir / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{info['status']}.png"
            cv2.imwrite(str(p), annotated)
            print("Saved", p)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
