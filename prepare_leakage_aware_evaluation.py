import csv
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path.cwd()
MANIFEST = ROOT / "datasets" / "dataset_manifest.csv"
OUT = ROOT / "reports" / "dataset_audit" / "evaluation"

OUT.mkdir(parents=True, exist_ok=True)

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows to write: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

rows = read_csv(MANIFEST)

if len(rows) != 640:
    raise SystemExit(f"STOP: Expected 640 manifest rows, found {len(rows)}.")

# Verify unique image paths
paths = [r["image_path"].replace("/", "\\").casefold() for r in rows]
if len(set(paths)) != len(paths):
    raise SystemExit("STOP: Duplicate image paths found in manifest.")

# Verify all confirmed groups stay within one split and label
groups = defaultdict(list)
for row in rows:
    sid = row.get("physical_sample_id", "").strip()
    if sid:
        groups[sid].append(row)

for sid, members in groups.items():
    splits = {r["split"].strip().casefold() for r in members}
    labels = {r["label"].strip().casefold() for r in members}
    if len(splits) != 1 or len(labels) != 1:
        raise SystemExit(f"STOP: Group {sid} crosses splits or labels.")

test_rows = [r for r in rows if r["split"].strip().casefold() == "test"]

if len(test_rows) != 200:
    raise SystemExit(f"STOP: Expected 200 test images, found {len(test_rows)}.")

# Standard evaluation: retain every test image
full_test = list(test_rows)

# Sample-aware evaluation:
# retain one deterministic representative from each confirmed test group;
# retain every test image whose sample identity is unconfirmed.
test_groups = defaultdict(list)
ungrouped = []

for row in test_rows:
    sid = row.get("physical_sample_id", "").strip()
    if sid:
        test_groups[sid].append(row)
    else:
        ungrouped.append(row)

representatives = []
excluded_repeats = []

for sid, members in sorted(test_groups.items()):
    members = sorted(
        members,
        key=lambda r: r["image_path"].replace("/", "\\").casefold()
    )
    representatives.append(members[0])
    excluded_repeats.extend(members[1:])

sample_aware_test = ungrouped + representatives
sample_aware_test.sort(key=lambda r: r["image_path"].casefold())
excluded_repeats.sort(key=lambda r: r["image_path"].casefold())

if len(sample_aware_test) + len(excluded_repeats) != len(test_rows):
    raise SystemExit("STOP: Test image accounting does not reconcile.")

# Save evaluation files
write_csv(OUT / "test_all_images.csv", full_test)
write_csv(OUT / "test_one_per_confirmed_sample.csv", sample_aware_test)

if excluded_repeats:
    write_csv(OUT / "test_confirmed_repeats_excluded.csv", excluded_repeats)

# Class counts
full_counts = Counter(r["label"] for r in full_test)
sample_counts = Counter(r["label"] for r in sample_aware_test)

report = [
    "PHARMASIGHT AI - LEAKAGE-AWARE EVALUATION PREPARATION",
    "=" * 58,
    f"Total manifest images: {len(rows)}",
    f"Total confirmed physical groups: {len(groups)}",
    f"Total test images: {len(full_test)}",
    f"Confirmed physical groups in test: {len(test_groups)}",
    f"Test images without confirmed sample IDs: {len(ungrouped)}",
    f"Test representatives retained: {len(representatives)}",
    f"Confirmed repeated test images excluded from secondary set: {len(excluded_repeats)}",
    f"Full test set size: {len(full_test)}",
    f"Sample-aware test set size: {len(sample_aware_test)}",
    "",
    "FULL TEST CLASS COUNTS:",
    str(dict(full_counts)),
    "",
    "SAMPLE-AWARE TEST CLASS COUNTS:",
    str(dict(sample_counts)),
    "",
    "FILES CREATED:",
    "test_all_images.csv",
    "test_one_per_confirmed_sample.csv",
    "test_confirmed_repeats_excluded.csv (if applicable)",
    "",
    "IMPORTANT:",
    "No source images were deleted, moved, or modified.",
    "The original train/validation/test assignments were not changed.",
    "The sample-aware set removes only confirmed repeated images within test.",
    "Unconfirmed physical identities remain included.",
    "This is a secondary sensitivity evaluation, not proof that all remaining",
    "images represent independent physical tablets.",
]

(OUT / "evaluation_protocol.txt").write_text(
    "\n".join(report) + "\n", encoding="utf-8"
)

print("\n========== EVALUATION PREPARATION ==========")
print("\nFull test set:", len(full_test))
print("Confirmed test groups:", len(test_groups))
print("Ungrouped test images:", len(ungrouped))
print("Representatives retained:", len(representatives))
print("Confirmed repeats excluded:", len(excluded_repeats))
print("Sample-aware test set:", len(sample_aware_test))

print("\nFull test class counts:", dict(full_counts))
print("Sample-aware class counts:", dict(sample_counts))

print("\nFiles saved in:")
print(OUT)

print("\nSUCCESS: Evaluation files prepared.")
