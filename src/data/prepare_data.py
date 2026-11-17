"""Phase 1 helper: check manifest completeness before any modelling."""
from pathlib import Path
import argparse
import pandas as pd

REQUIRED = ["image_path", "task", "source", "split", "label", "licence"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, default=Path("datasets/dataset_manifest.csv"))
    args = ap.parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    df = pd.read_csv(args.manifest).fillna("")
    missing_cols = [c for c in REQUIRED if c not in df.columns]
    if missing_cols:
        raise SystemExit(f"Missing required columns: {missing_cols}")
    print("Rows:", len(df))
    for col in ["task", "source", "split", "label", "licence"]:
        print(f"\n{col}:\n{df[col].value_counts(dropna=False).to_string()}")
    incomplete = (df[["source", "split", "label", "licence"]] == "").any(axis=1)
    print(f"\nRows requiring metadata review: {int(incomplete.sum())}")
    print("No train/validation/test files are generated automatically: define leakage-safe splits first.")

if __name__ == "__main__":
    main()
