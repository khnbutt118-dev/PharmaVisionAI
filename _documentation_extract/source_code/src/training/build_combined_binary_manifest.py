from pathlib import Path
from datetime import datetime
import pandas as pd
import json

ROOT = Path(".")
PHARMA_FILE = ROOT / "datasets/dataset_manifest.csv"
PILLQC_FILE = ROOT / "reports/training/pillqc_multiclass/pillqc_multiclass_split_manifest.csv"

OUTPUT = ROOT / "datasets/combined_binary_manifest.csv"
REPORT_DIR = ROOT / "reports/dataset_audit/combined"

COMMON_COLUMNS = [
    "image_path",
    "task",
    "source",
    "dataset",
    "physical_sample_id",
    "capture_session",
    "split",
    "label",
    "defect_type",
    "licence",
    "notes",
]

EXTRA_COLUMNS = [
    "binary_label",
    "subtype_label",
    "source_manifest",
]

def load_and_prepare(path, source_name):
    if not path.exists():
        raise FileNotFoundError(f"Missing manifest: {path}")

    df = pd.read_csv(path)

    required = ["image_path", "label", "split", "defect_type"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"{source_name} missing columns: {missing}")

    df = df.copy()

    # Only individual tablet images with known binary labels.
    df = df[
        (df["task"] == "tablet") &
        (df["label"].isin(["normal", "defective"]))
    ].copy()

    df["source"] = source_name

    # Fill missing dataset name without changing the original source.
    if "dataset" not in df.columns:
        df["dataset"] = source_name
    else:
        df["dataset"] = df["dataset"].fillna(source_name)

    df["source_manifest"] = str(path)
    df["binary_label"] = df["label"]
    df["subtype_label"] = df["defect_type"]

    return df

print("=" * 65)
print("PHARMASIGHT AI - COMBINED DATASET BUILDER")
print("=" * 65)

pharma = load_and_prepare(PHARMA_FILE, "PharmaVision")
pillqc = load_and_prepare(PILLQC_FILE, "PillQC")

combined = pd.concat([pharma, pillqc], ignore_index=True)

# Keep only the agreed columns.
for col in COMMON_COLUMNS + EXTRA_COLUMNS:
    if col not in combined.columns:
        combined[col] = pd.NA

combined = combined[COMMON_COLUMNS + EXTRA_COLUMNS]

# Validate labels and split assignments.
if not combined["label"].isin(["normal", "defective"]).all():
    raise ValueError("Unexpected binary label found.")

if not combined["split"].isin(["train", "validation", "test"]).all():
    raise ValueError("Missing or invalid split assignment found.")

# Validate every image path.
missing_paths = []

for value in combined["image_path"]:
    if pd.isna(value) or not Path(str(value)).is_file():
        missing_paths.append(str(value))

if missing_paths:
    print("\nMissing image paths:", len(missing_paths))
    for item in missing_paths[:20]:
        print(item)
    raise FileNotFoundError("Fix missing image paths before training.")

# Check duplicate manifest paths.
normalized_paths = (
    combined["image_path"]
    .astype(str)
    .str.replace("/", "\\", regex=False)
    .str.lower()
)

if normalized_paths.duplicated().any():
    duplicates = combined[normalized_paths.duplicated(keep=False)]
    print(duplicates[["image_path", "source", "split"]].to_string(index=False))
    raise ValueError("Duplicate image paths detected.")

# Expected totals from our verified manifests.
expected = {
    ("PharmaVision", "train"): 380,
    ("PharmaVision", "validation"): 60,
    ("PharmaVision", "test"): 200,
    ("PillQC", "train"): 229,
    ("PillQC", "validation"): 51,
    ("PillQC", "test"): 50,
}

actual = combined.groupby(["source", "split"]).size().to_dict()

for key, expected_count in expected.items():
    actual_count = actual.get(key, 0)
    if actual_count != expected_count:
        raise ValueError(
            f"Count mismatch for {key}: "
            f"expected {expected_count}, got {actual_count}"
        )

# Create output folders.
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

# Preserve any previous output.
if OUTPUT.exists():
    backup = OUTPUT.with_name(
        f"combined_binary_manifest_backup_"
        f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    )
    OUTPUT.replace(backup)
    print("Previous manifest backed up:", backup)

combined.to_csv(OUTPUT, index=False)

# Create source/split/label summary.
summary = (
    combined.groupby(
        ["source", "split", "label", "defect_type"],
        dropna=False
    )
    .size()
    .reset_index(name="image_count")
)

summary_path = REPORT_DIR / "combined_dataset_summary.csv"
summary.to_csv(summary_path, index=False)

# Create a compact audit report.
audit = {
    "total_images": int(len(combined)),
    "sources": combined["source"].value_counts().to_dict(),
    "binary_labels": combined["label"].value_counts().to_dict(),
    "splits": combined["split"].value_counts().to_dict(),
    "source_split_counts": {
        f"{source}_{split}": int(count)
        for (source, split), count in actual.items()
    },
    "missing_image_paths": 0,
    "duplicate_manifest_paths": 0,
    "output_manifest": str(OUTPUT),
}

audit_path = REPORT_DIR / "combined_dataset_audit.json"
with open(audit_path, "w", encoding="utf-8") as f:
    json.dump(audit, f, indent=4)

print("\n========== BUILD SUCCESSFUL ==========")
print("Total images:", len(combined))
print("\nBinary label counts:")
print(combined["label"].value_counts().to_string())

print("\nSplit counts:")
print(combined["split"].value_counts().to_string())

print("\nSource and split counts:")
print(
    combined.groupby(["source", "split"])
    .size()
    .to_string()
)

print("\nDefect subtype counts:")
print(combined["defect_type"].value_counts().to_string())

print("\nMissing images: 0")
print("Duplicate paths: 0")

print("\nCombined manifest:", OUTPUT)
print("Summary report:", summary_path)
print("Audit report:", audit_path)
