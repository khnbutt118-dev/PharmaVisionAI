import csv
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ROOT = Path.cwd()

DECISIONS = ROOT / "reports/dataset_audit/duplicate_review_decisions_updated.csv"
PAIRS = ROOT / "reports/dataset_audit/duplicate_review_template.csv"
MANIFEST = ROOT / "datasets/dataset_manifest.csv"

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))

def norm(path):
    return str(path).replace("/", "\\").strip().casefold()

# Load files
decisions = read_csv(DECISIONS)
pairs = read_csv(PAIRS)
manifest = read_csv(MANIFEST)

if not manifest:
    raise SystemExit("ERROR: Manifest is empty.")

# Validate required columns
required_manifest = {
    "image_path", "physical_sample_id", "split", "label"
}
if not required_manifest.issubset(manifest[0].keys()):
    raise SystemExit("ERROR: Required manifest columns are missing.")

if not {"pair_number", "review_decision"}.issubset(decisions[0].keys()):
    raise SystemExit("ERROR: Decisions file has unexpected columns.")

if not {"pair_number", "image_a", "image_b"}.issubset(pairs[0].keys()):
    raise SystemExit("ERROR: Pair template has unexpected columns.")

# Refuse to overwrite existing physical sample IDs
existing = [r["image_path"] for r in manifest
            if r.get("physical_sample_id", "").strip()]
if existing:
    raise SystemExit(
        f"STOP: {len(existing)} manifest rows already have physical_sample_id. "
        "No changes made."
    )

# Build pair-number lookup
pair_lookup = {}
for row in pairs:
    number = row["pair_number"].strip()
    if number in pair_lookup:
        raise SystemExit(f"ERROR: Duplicate pair number in template: {number}")
    pair_lookup[number] = row

decision_lookup = {}
for row in decisions:
    number = row["pair_number"].strip()
    if number in decision_lookup:
        raise SystemExit(f"ERROR: Duplicate decision number: {number}")
    decision_lookup[number] = row["review_decision"].strip()

if set(pair_lookup) != set(decision_lookup):
    raise SystemExit(
        "ERROR: Pair numbers do not match between decisions and template."
    )

# Build manifest lookup
manifest_lookup = {}
for row in manifest:
    key = norm(row["image_path"])
    if key in manifest_lookup:
        raise SystemExit(f"ERROR: Duplicate manifest image path: {key}")
    manifest_lookup[key] = row

# Union-find for connected physical-image groups
parent = {}

def find(x):
    parent.setdefault(x, x)
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    ra, rb = find(a), find(b)
    if ra != rb:
        parent[max(ra, rb)] = min(ra, rb)

same_pairs = 0
different_pairs = 0
missing_images = []

for number, decision in decision_lookup.items():
    pair = pair_lookup[number]
    a = norm(pair["image_a"])
    b = norm(pair["image_b"])

    if a not in manifest_lookup:
        missing_images.append(pair["image_a"])
    if b not in manifest_lookup:
        missing_images.append(pair["image_b"])

    if decision == "same_physical_image":
        same_pairs += 1
        union(a, b)
    elif decision == "different_images":
        different_pairs += 1
    else:
        raise SystemExit(
            f"ERROR: Unresolved decision for pair {number}: {decision}"
        )

if missing_images:
    raise SystemExit(
        "ERROR: Some pair images are missing from the manifest:\n" +
        "\n".join(missing_images[:20])
    )

if same_pairs != 40 or different_pairs != 187:
    raise SystemExit(
        f"ERROR: Expected 40 same and 187 different decisions; "
        f"found {same_pairs} and {different_pairs}. No changes made."
    )

# Collect connected components
components = defaultdict(list)
for image in parent:
    components[find(image)].append(image)

groups = sorted(
    [sorted(images) for images in components.values()],
    key=lambda group: group[0]
)

# Validate each group against manifest split and label
for group in groups:
    records = [manifest_lookup[path] for path in group]
    splits = {r["split"].strip().casefold() for r in records}
    labels = {r["label"].strip().casefold() for r in records}

    if len(splits) != 1:
        raise SystemExit(
            "SAFETY STOP: A confirmed physical group crosses splits:\n" +
            "\n".join(group)
        )

    if len(labels) != 1:
        raise SystemExit(
            "SAFETY STOP: A confirmed physical group crosses labels:\n" +
            "\n".join(group)
        )

# Assign IDs to confirmed groups only
for index, group in enumerate(groups, start=1):
    sample_id = f"PV_SAMPLE_{index:03d}"
    for path in group:
        manifest_lookup[path]["physical_sample_id"] = sample_id

# Backup before writing
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = MANIFEST.with_name(
    f"dataset_manifest.backup_before_sample_groups_{timestamp}.csv"
)

with MANIFEST.open("rb") as source:
    backup.write_bytes(source.read())

# Write updated manifest
with MANIFEST.open("w", encoding="utf-8-sig", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(manifest[0].keys()))
    writer.writeheader()
    writer.writerows(manifest)

# Reload and verify
verified = read_csv(MANIFEST)
assigned = [r for r in verified if r["physical_sample_id"].strip()]
unique_ids = sorted({r["physical_sample_id"] for r in assigned})

print("\n========== PHYSICAL SAMPLE GROUPING REPORT ==========")
print(f"Reviewed pairs: {len(decisions)}")
print(f"Same-image pairs: {same_pairs}")
print(f"Different-image pairs: {different_pairs}")
print(f"Confirmed physical groups: {len(groups)}")
print(f"Images assigned to groups: {len(assigned)}")
print(f"Unique physical sample IDs: {len(unique_ids)}")
print(f"Images left unassigned: {len(verified) - len(assigned)}")
print(f"Manifest rows after update: {len(verified)}")
print(f"Backup created: {backup.name}")

print("\n========== GROUP DETAILS ==========")
for index, group in enumerate(groups, start=1):
    print(f"\nPV_SAMPLE_{index:03d} ({len(group)} images)")
    for path in group:
        print("  " + manifest_lookup[path]["image_path"])

print("\nSUCCESS: Manifest updated and verification completed.")
