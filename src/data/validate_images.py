"""Validate image readability and report dimensions; does not alter source files."""
import argparse
from pathlib import Path
import cv2
import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

def inspect(root: Path, recursive: bool = True) -> pd.DataFrame:
    paths = root.rglob("*") if recursive else root.glob("*")
    rows = []
    for path in sorted(p for p in paths if p.is_file() and p.suffix.lower() in EXTS):
        image = cv2.imread(str(path))
        if image is None:
            rows.append({"image_path": str(path), "readable": False,
                         "width": None, "height": None, "channels": None})
        else:
            h, w = image.shape[:2]
            rows.append({"image_path": str(path), "readable": True,
                         "width": w, "height": h,
                         "channels": 1 if image.ndim == 2 else image.shape[2]})
    return pd.DataFrame(rows, columns=[
        "image_path", "readable", "width", "height", "channels"
    ])

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/dataset_audit/image_validation.csv"))
    parser.add_argument("--recursive", action="store_true")
    args = parser.parse_args()
    if not args.root.exists():
        raise SystemExit(f"Folder does not exist: {args.root}")
    df = inspect(args.root, args.recursive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Images found: {len(df)}")
    if len(df):
        print(f"Readable: {int(df.readable.sum())}; unreadable: {int((~df.readable).sum())}")
        print(f"Report saved: {args.output}")
    else:
        print("No supported image files found. Check the dataset folder and extraction.")

if __name__ == "__main__":
    main()
