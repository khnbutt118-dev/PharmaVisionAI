from pathlib import Path
import pandas as pd

project = Path(".").resolve()
search_root = project / "datasets" / "external"

# Find the actual PillQC image folder without assuming its nesting.
candidates = [
    p.parent for p in search_root.rglob("normal")
    if p.is_dir()
    and p.name.lower() == "normal"
    and p.parent.name.lower() == "images"
    and (p.parent / "dirt").is_dir()
    and (p.parent / "chip").is_dir()
]

if len(candidates) != 1:
    print("ERROR: Expected exactly one PillQC images folder.")
    print("Found:", len(candidates))
    for p in candidates:
        print(p)
    raise SystemExit(1)

images_root = candidates[0]
dataset_root = images_root.parent
extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

rows = []

categories = {
    "normal": {
        "label": "normal",
        "defect_type": "none",
        "sample_id": "PILLQC_NORMAL_SHARED_SAMPLE",
        "notes": "Provisional shared sample group: user reports repeated views of the same physical tablet."
    },
    "dirt": {
        "label": "defective",
        "defect_type": "contamination",
        "sample_id": "PILLQC_DIRT_SHARED_SAMPLE",
        "notes": "Provisional shared sample group: user reports repeated views of the same tablet with varying dirt patterns."
    },
    "chip": {
        "label": "defective",
        "defect_type": "chipping",
        "sample_id": "",
        "notes": "Visible broken edges or missing corners; physical sample identity not established."
    }
}

for category, info in categories.items():
    folder = images_root / category

    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in extensions:
            continue

        rows.append({
            "image_path": str(path.relative_to(project)).replace("/", "\\"),
            "task": "tablet",
            "source": "PillQC",
            "dataset": "PillQC",
            "physical_sample_id": info["sample_id"],
            "capture_session": "",
            "split": "",
            "label": info["label"],
            "defect_type": info["defect_type"],
            "licence": "XSLA - terms require verification",
            "notes": info["notes"]
        })

# Record group photographs separately. They are NOT individual tablet samples.
source_folder = dataset_root / "sourceImages"

for path in sorted(source_folder.iterdir()):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    rows.append({
        "image_path": str(path.relative_to(project)).replace("/", "\\"),
        "task": "source_group_photo",
        "source": "PillQC",
        "dataset": "PillQC",
        "physical_sample_id": "",
        "capture_session": "",
        "split": "",
        "label": "unresolved",
        "defect_type": "unresolved",
        "licence": "XSLA - terms require verification",
        "notes": "Group photograph; not approved for individual-tablet classifier training."
    })

columns = [
    "image_path", "task", "source", "dataset",
    "physical_sample_id", "capture_session", "split",
    "label", "defect_type", "licence", "notes"
]

df = pd.DataFrame(rows, columns=columns)

output = project / "datasets" / "external" / "pillQC_manifest.csv"
output.parent.mkdir(parents=True, exist_ok=True)

if output.exists():
    print("ERROR: Manifest already exists. No file was overwritten:")
    print(output)
    raise SystemExit(1)

df.to_csv(output, index=False, encoding="utf-8-sig")

print("\n========== PILLQC MANIFEST CREATED ==========")
print("Image folder:", images_root)
print("Manifest:", output)
print("\nTask counts:")
print(df["task"].value_counts().to_string())
print("\nLabels:")
print(df.groupby(["task", "label", "defect_type"]).size().to_string())
print("\nRows:", len(df))
print("Unique image paths:", df["image_path"].nunique())
print("Splits assigned: NONE (intentionally)")
print("\nExisting dataset manifest was not changed.")
