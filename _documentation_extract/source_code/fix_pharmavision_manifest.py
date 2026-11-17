#!/usr/bin/env python3
"""
Safely repair the PharmaSight-AI tablet dataset manifest.

Reads the existing manifest, derives split and class only from the exact
PharmaVision folder names in each image path, creates a timestamped backup,
and then writes the repaired manifest. It does not move, rename, resize,
delete, or modify any images.
"""

import csv
import sys
from datetime import datetime
from pathlib import Path, PurePosixPath

PROJECT_ROOT = Path(__file__).resolve().parent
MANIFEST = PROJECT_ROOT / "datasets" / "dataset_manifest.csv"

SPLITS = {"train", "validation", "test"}
LABELS = {"normal", "defective"}
EXPECTED_DATASET_FOLDER = "PharmaVision dataset"
LICENSE_VALUE = "CC BY-SA 4.0"


def normalized_parts(value: str):
    # CSV paths may have been written with Windows or POSIX separators.
    return PurePosixPath(str(value).replace("\\", "/")).parts


def derive_metadata(image_path: str):
    parts = normalized_parts(image_path)

    # Restrict the mapping to the expected PharmaVision dataset subtree.
    try:
        dataset_index = parts.index(EXPECTED_DATASET_FOLDER)
    except ValueError:
        return None

    after_dataset = parts[dataset_index + 1:]
    split = next((part for part in after_dataset if part in SPLITS), None)
    label = next((part for part in after_dataset if part in LABELS), None)

    if split is None or label is None:
        return None

    # Require the split folder to appear before the class folder.
    if after_dataset.index(split) > after_dataset.index(label):
        return None

    return split, label


def main():
    if not MANIFEST.exists():
        print(f"ERROR: Manifest not found:\n{MANIFEST}")
        print("Run this script from the inner PharmaSight-AI project folder.")
        sys.exit(1)

    with MANIFEST.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        print("ERROR: Manifest is empty. Nothing was changed.")
        sys.exit(1)

    required = {"image_path", "split", "label"}
    missing_columns = required - set(fieldnames)
    if missing_columns:
        print("ERROR: Missing required columns:", ", ".join(sorted(missing_columns)))
        print("Nothing was changed.")
        sys.exit(1)

    if "licence" not in fieldnames:
        fieldnames.append("licence")
    if "defect_type" not in fieldnames:
        fieldnames.append("defect_type")
    if "notes" not in fieldnames:
        fieldnames.append("notes")

    unmatched = []
    for row_number, row in enumerate(rows, start=2):
        metadata = derive_metadata(row.get("image_path", ""))
        if metadata is None:
            unmatched.append((row_number, row.get("image_path", "")))
            continue

        split, label = metadata
        row["split"] = split
        row["label"] = label

        # Only fill licence when it is blank or the earlier placeholder.
        if not (row.get("licence") or "").strip() or \
           (row.get("licence") or "").strip().lower() in {"verify", "unknown", "tbd"}:
            row["licence"] = LICENSE_VALUE

        # Do not invent a defect subtype.
        if label == "defective" and not (row.get("defect_type") or "").strip():
            row["defect_type"] = "unresolved"
        elif label == "normal" and not (row.get("defect_type") or "").strip():
            row["defect_type"] = "none"

        note = (row.get("notes") or "").strip()
        audit_note = "split and label derived from PharmaVision folder path; verify with dataset documentation"
        if audit_note not in note:
            row["notes"] = f"{note}; {audit_note}".strip("; ")

    if unmatched:
        print(f"ERROR: Could not safely identify split/class for {len(unmatched)} row(s).")
        for row_number, path in unmatched[:10]:
            print(f"  CSV row {row_number}: {path}")
        if len(unmatched) > 10:
            print(f"  ... plus {len(unmatched) - 10} more")
        print("Nothing was changed. Check the folder paths before retrying.")
        sys.exit(1)

    # Sanity check expected current dataset size and balanced folder labels.
    counts = {}
    for row in rows:
        key = (row["split"], row["label"])
        counts[key] = counts.get(key, 0) + 1

    if len(rows) != 640:
        print(f"WARNING: Found {len(rows)} manifest rows, not the previously reported 640.")
        print("The script will still proceed because every row had a valid folder mapping.")

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup = MANIFEST.with_name(f"dataset_manifest.backup_{stamp}.csv")
    backup.write_bytes(MANIFEST.read_bytes())

    with MANIFEST.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    print("SUCCESS: Manifest repaired. Image files were not modified.")
    print(f"Rows updated: {len(rows)}")
    print(f"Backup saved: {backup}")
    print("\nCounts by split and label:")
    for split in ("train", "validation", "test"):
        for label in ("normal", "defective"):
            print(f"  {split:10s} {label:10s}: {counts.get((split, label), 0)}")
    print("\nImportant:")
    print("- Folder-derived labels still need confirmation against dataset documentation.")
    print("- 'unresolved' is not a confirmed defect subtype.")
    print("- physical_sample_id and capture_session remain blank; no values were invented.")
    print("- The 227 near-duplicate candidates still need visual review before training.")


if __name__ == "__main__":
    main()
