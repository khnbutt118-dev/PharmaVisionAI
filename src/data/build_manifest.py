"""Create an image inventory. Labels/licences are intentionally left for human review."""
import argparse
from pathlib import Path
import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
COLUMNS = ["image_path", "task", "source", "dataset", "physical_sample_id",
           "capture_session", "split", "label", "defect_type", "licence", "notes"]

def build(root: Path, task: str) -> pd.DataFrame:
    rows = []
    for p in sorted(x for x in root.rglob("*") if x.is_file() and x.suffix.lower() in EXTS):
        try:
            rel = p.relative_to(root)
            source = rel.parts[0] if len(rel.parts) > 1 else "unsorted_source"
        except ValueError:
            source = "unknown"
        rows.append({
            "image_path": str(p), "task": task, "source": source,
            "dataset": "", "physical_sample_id": "", "capture_session": "",
            "split": "", "label": "", "defect_type": "", "licence": "verify",
            "notes": "Review label, source, permissions and sample identity manually."
        })
    return pd.DataFrame(rows, columns=COLUMNS)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--output", type=Path, default=Path("datasets/dataset_manifest.csv"))
    ap.add_argument("--task", default="tablet", choices=["tablet", "blister", "ppe"])
    args = ap.parse_args()
    if not args.root.exists():
        raise SystemExit(f"Folder does not exist: {args.root}")
    df = build(args.root, args.task)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Preserve existing entries for other tasks, replacing only this task's inventory.
    if args.output.exists():
        old = pd.read_csv(args.output).fillna("")
        if "task" in old.columns:
            old = old[old.task != args.task]
        df = pd.concat([old, df], ignore_index=True)
    df.to_csv(args.output, index=False)
    print(f"Manifest rows for {args.task}: {len(df[df.task == args.task])}")
    print(f"Saved: {args.output}")
    print("Important: this script does not infer or verify labels or dataset licences.")

if __name__ == "__main__":
    main()
