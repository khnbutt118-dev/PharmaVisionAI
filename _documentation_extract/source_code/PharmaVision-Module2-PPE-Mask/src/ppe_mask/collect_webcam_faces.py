"""Collect YOUR OWN labelled webcam frames (the real test domain).

Run:  python src/ppe_mask/collect_webcam_faces.py --session s1_desk_daylight
Controls: SPACE = freeze frame -> drag a box around the FACE -> press ENTER
          then press  M = with_mask, N = without_mask, I = mask_weared_incorrect, ESC = discard
          Q = quit
Saves YOLO labels to datasets/ppe_mask/webcam/<session>/{images,labels}.
Use a DIFFERENT session name for each lighting/background/distance/person setup, and keep at least
one session out of training. Only record people who have agreed to it.
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
from src.ppe_mask.gate import CLASS_NAMES  # noqa: E402

KEY_TO_CLASS = {ord("m"): 0, ord("n"): 1, ord("i"): 2}


def yolo_line(cls_id: int, x: int, y: int, w: int, h: int, img_w: int, img_h: int) -> str:
    xc, yc = (x + w / 2) / img_w, (y + h / 2) / img_h
    return f"{cls_id} {xc:.6f} {yc:.6f} {w / img_w:.6f} {h / img_h:.6f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True)
    ap.add_argument("--root", type=Path, default=Path("datasets/ppe_mask/webcam"))
    ap.add_argument("--camera", type=int, default=0)
    a = ap.parse_args()
    img_dir, lab_dir = a.root / a.session / "images", a.root / a.session / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lab_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(a.camera, cv2.CAP_DSHOW) if os.name == "nt" else cv2.VideoCapture(a.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open camera.")
    counts = {n: 0 for n in CLASS_NAMES}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        view = frame.copy()
        cv2.putText(view, f"session {a.session} | M:{counts['with_mask']} N:{counts['without_mask']} I:{counts['mask_weared_incorrect']}",
                    (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.putText(view, "SPACE = capture, Q = quit", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        cv2.imshow("collect mask frames", view)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == 32:
            x, y, w, h = cv2.selectROI("collect mask frames", frame, showCrosshair=False)
            if w < 10 or h < 10:
                continue
            prompt = frame.copy()
            cv2.rectangle(prompt, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(prompt, "M=mask  N=no mask  I=incorrect  ESC=discard", (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("collect mask frames", prompt)
            while True:
                kk = cv2.waitKey(0) & 0xFF
                if kk == 27 or kk in KEY_TO_CLASS:
                    break
            if kk in KEY_TO_CLASS:
                cid = KEY_TO_CLASS[kk]
                stamp = time.strftime("%Y%m%d_%H%M%S") + f"_{int(time.time() * 1000) % 1000:03d}"
                cv2.imwrite(str(img_dir / f"{a.session}_{stamp}.jpg"), frame)
                (lab_dir / f"{a.session}_{stamp}.txt").write_text(
                    yolo_line(cid, x, y, w, h, frame.shape[1], frame.shape[0]) + "\n")
                counts[CLASS_NAMES[cid]] += 1
    cap.release()
    cv2.destroyAllWindows()
    print("Saved:", counts, "->", a.root / a.session)


if __name__ == "__main__":
    main()
