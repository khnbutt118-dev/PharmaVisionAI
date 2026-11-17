"""Convert the Kaggle 'Face Mask Detection' dataset (PASCAL VOC XML) to YOLO format.

Expected raw layout (download from https://www.kaggle.com/datasets/andrewmvd/face-mask-detection):
    datasets/ppe_mask/raw/kaggle_face_mask/images/*.png
    datasets/ppe_mask/raw/kaggle_face_mask/annotations/*.xml

Output (default datasets/ppe_mask/prepared):
    images/{train,val,test}, labels/{train,val,test}, data.yaml, prepare_report.json

Notes
  * Exact duplicate images (same SHA-256) are dropped before splitting so they cannot leak across splits.
    Near-duplicates are NOT detected here; run src/data/find_duplicates.py if you have it.
  * Your own webcam sessions (datasets/ppe_mask/webcam/<session>/{images,labels}) can be added to TRAIN only,
    by session name. Keep at least one session out of training as your honest test set.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import random
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.ppe_mask.gate import CLASS_NAMES  # noqa: E402

CLASS_ID = {n: i for i, n in enumerate(CLASS_NAMES)}
IMG_EXTS = [".png", ".jpg", ".jpeg", ".bmp"]


def parse_voc(xml_path: Path):
    root = ET.parse(xml_path).getroot()
    fname = root.findtext("filename") or ""
    objs = []
    for o in root.findall("object"):
        name = (o.findtext("name") or "").strip()
        bb = o.find("bndbox")
        if bb is None:
            continue
        vals = [float(bb.findtext(k)) for k in ("xmin", "ymin", "xmax", "ymax")]
        objs.append((name, *vals))
    return fname, objs


def to_yolo(box, w, h):
    xmin, ymin, xmax, ymax = box
    xmin, xmax = max(0.0, xmin), min(float(w), xmax)
    ymin, ymax = max(0.0, ymin), min(float(h), ymax)
    bw, bh = xmax - xmin, ymax - ymin
    if bw <= 1 or bh <= 1:
        return None
    return ((xmin + xmax) / 2 / w, (ymin + ymax) / 2 / h, bw / w, bh / h)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_image(images_dir: Path, stem: str, hint: str):
    cands = []
    if hint:
        cands.append(images_dir / hint)
    cands += [images_dir / (stem + e) for e in IMG_EXTS]
    for c in cands:
        if c.exists():
            return c
    return None


def convert(raw_dir: Path, out_dir: Path, seed=42, val_frac=0.15, test_frac=0.15,
            webcam_root: Path | None = None, train_sessions=()):
    import cv2  # local import keeps the module importable for tests that never touch images
    images_dir, ann_dir = raw_dir / "images", raw_dir / "annotations"
    if not images_dir.is_dir() or not ann_dir.is_dir():
        raise SystemExit(f"Expected {images_dir} and {ann_dir}. See docs/PPE_MASK_MODULE.md for the layout.")
    report = {"skipped_missing_image": 0, "skipped_unreadable_image": 0, "skipped_unknown_class_objects": 0,
              "skipped_degenerate_boxes": 0, "duplicate_images_dropped": 0, "images_without_objects": 0}
    items, seen = [], {}
    for xml in sorted(ann_dir.glob("*.xml")):
        fname, objs = parse_voc(xml)
        img = find_image(images_dir, xml.stem, fname)
        if img is None:
            report["skipped_missing_image"] += 1
            continue
        im = cv2.imread(str(img))
        if im is None:
            report["skipped_unreadable_image"] += 1
            continue
        h, w = im.shape[:2]
        digest = sha256_file(img)
        if digest in seen:
            report["duplicate_images_dropped"] += 1
            continue
        seen[digest] = img.name
        lines = []
        for name, *box in objs:
            if name not in CLASS_ID:
                report["skipped_unknown_class_objects"] += 1
                continue
            y = to_yolo(box, w, h)
            if y is None:
                report["skipped_degenerate_boxes"] += 1
                continue
            lines.append(f"{CLASS_ID[name]} " + " ".join(f"{v:.6f}" for v in y))
        if not lines:
            report["images_without_objects"] += 1
            continue
        items.append((img, lines))

    rng = random.Random(seed)
    order = list(range(len(items)))
    rng.shuffle(order)
    n = len(order)
    n_test, n_val = int(round(n * test_frac)), int(round(n * val_frac))
    split_of = {}
    for k, idx in enumerate(order):
        split_of[idx] = "test" if k < n_test else ("val" if k < n_test + n_val else "train")

    if out_dir.exists():
        shutil.rmtree(out_dir)
    for s in ("train", "val", "test"):
        (out_dir / "images" / s).mkdir(parents=True)
        (out_dir / "labels" / s).mkdir(parents=True)
    counts = {s: {"images": 0, **{c: 0 for c in CLASS_NAMES}} for s in ("train", "val", "test")}

    def emit(split, img_path: Path, lines, prefix=""):
        stem = prefix + img_path.stem
        shutil.copy2(img_path, out_dir / "images" / split / (stem + img_path.suffix.lower()))
        (out_dir / "labels" / split / (stem + ".txt")).write_text("\n".join(lines) + "\n")
        counts[split]["images"] += 1
        for ln in lines:
            counts[split][CLASS_NAMES[int(ln.split()[0])]] += 1

    for idx, (img, lines) in enumerate(items):
        emit(split_of[idx], img, lines)

    webcam_added = 0
    if webcam_root is not None:
        for sess in train_sessions:
            img_d, lab_d = webcam_root / sess / "images", webcam_root / sess / "labels"
            for img in sorted(img_d.glob("*")):
                if img.suffix.lower() not in IMG_EXTS:
                    continue
                lab = lab_d / (img.stem + ".txt")
                if not lab.exists():
                    continue
                lines = [l for l in lab.read_text().splitlines() if l.strip()]
                if lines:
                    emit("train", img, lines, prefix=f"webcam_{sess}_")
                    webcam_added += 1

    import yaml
    data = {"path": out_dir.resolve().as_posix(), "train": "images/train", "val": "images/val",
            "test": "images/test", "names": {i: n for i, n in enumerate(CLASS_NAMES)}}
    (out_dir / "data.yaml").write_text(yaml.safe_dump(data, sort_keys=False))
    report.update({"seed": seed, "images_used": n, "split_counts": counts, "webcam_images_added_to_train": webcam_added,
                   "note": "Random image-level split of a public dataset; near-duplicates and same-scene images may still leak across splits."})
    (out_dir / "prepare_report.json").write_text(json.dumps(report, indent=2))
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", type=Path, default=Path("datasets/ppe_mask/raw/kaggle_face_mask"))
    ap.add_argument("--out", type=Path, default=Path("datasets/ppe_mask/prepared"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--val", type=float, default=0.15)
    ap.add_argument("--test", type=float, default=0.15)
    ap.add_argument("--webcam-root", type=Path, default=Path("datasets/ppe_mask/webcam"))
    ap.add_argument("--train-sessions", nargs="*", default=[], help="webcam session folders to add to TRAIN only")
    a = ap.parse_args()
    rep = convert(a.raw, a.out, a.seed, a.val, a.test, a.webcam_root, a.train_sessions)
    print(json.dumps(rep, indent=2))
    print(f"\nWrote {a.out / 'data.yaml'}")


if __name__ == "__main__":
    main()
