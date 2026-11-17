"""Operator-gate evaluation: the error rates that matter for a mask check.

For every image in a YOLO-format folder, the TRUTH is 'compliant' only if it has at least one labelled
face and ALL labelled faces are with_mask. The PREDICTION is 'compliant' only if the gate logic returns COMPLIANT
(UNCERTAIN and NO_FACE count as NOT compliant).
  false_accept = truth NOT compliant, predicted compliant   (an unmasked operator let through: the dangerous error)
  false_reject = truth compliant, predicted NOT compliant   (annoying but safe)

Examples:
  python src/ppe_mask/evaluate_gate.py --images datasets/ppe_mask/prepared/images/test --labels datasets/ppe_mask/prepared/labels/test
  python src/ppe_mask/evaluate_gate.py --images datasets/ppe_mask/webcam/s3/images --labels datasets/ppe_mask/webcam/s3/labels
Single frames only; it does not measure the temporal gate. Use a person/session NOT used in training.
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ppe_mask.gate import CLASS_NAMES, COMPLIANT, GateConfig, frame_status  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def truth_compliant(label_path: Path) -> bool | None:
    if not label_path.exists():
        return None
    ids = [int(l.split()[0]) for l in label_path.read_text().splitlines() if l.strip()]
    if not ids:
        return None
    return all(CLASS_NAMES[i] == "with_mask" for i in ids)


def evaluate(images_dir: Path, labels_dir: Path, predict_fn, cfg: GateConfig = GateConfig(), frame_area_fn=None):
    """predict_fn(path) -> (list[Detection], frame_area). Returns (summary, rows)."""
    rows, tp = [], dict(tn=0, tp=0, fa=0, fr=0)
    status_counts = {}
    for img in sorted(p for p in images_dir.iterdir() if p.suffix.lower() in EXTS):
        truth = truth_compliant(labels_dir / (img.stem + ".txt"))
        if truth is None:
            continue
        dets, area = predict_fn(img)
        status = frame_status(dets, cfg, frame_area=area)
        pred = status == COMPLIANT
        status_counts[status] = status_counts.get(status, 0) + 1
        if truth and pred:
            tp["tp"] += 1
        elif truth and not pred:
            tp["fr"] += 1
        elif not truth and pred:
            tp["fa"] += 1
        else:
            tp["tn"] += 1
        rows.append({"image": img.name, "truth_compliant": truth, "status": status, "predicted_compliant": pred})
    n_bad = tp["fa"] + tp["tn"]
    n_good = tp["tp"] + tp["fr"]
    summary = {"images": len(rows), "truth_compliant": n_good, "truth_not_compliant": n_bad,
               "false_accepts": tp["fa"], "false_rejects": tp["fr"],
               "false_accept_rate": (tp["fa"] / n_bad) if n_bad else None,
               "false_reject_rate": (tp["fr"] / n_good) if n_good else None,
               "status_counts": status_counts,
               "config": cfg.__dict__}
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default="runs/ppe_mask/mask_yolov8n/weights/best.pt")
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--labels", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--out", type=Path, default=Path("reports/ppe_mask/gate_eval"))
    a = ap.parse_args()
    from src.ppe_mask.mask_module import MaskComplianceModule
    import cv2
    mod = MaskComplianceModule(a.weights, conf=a.conf, imgsz=a.imgsz)

    def predict(path):
        im = cv2.imread(str(path))
        h, w = im.shape[:2]
        return mod.detect(im), float(h * w)

    summary, rows = evaluate(a.images, a.labels, predict, mod.cfg)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "gate_summary.json").write_text(json.dumps(summary, indent=2))
    with open(a.out / "gate_rows.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["image", "truth_compliant", "status", "predicted_compliant"])
        w.writeheader()
        w.writerows(rows)
    print(json.dumps(summary, indent=2))
    print(f"\nSaved to {a.out}. Look at every false accept image by eye.")


if __name__ == "__main__":
    main()
