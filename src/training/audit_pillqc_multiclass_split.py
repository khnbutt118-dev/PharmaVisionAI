from pathlib import Path
from itertools import combinations
from hashlib import sha256
import csv

import pandas as pd
from PIL import Image

MANIFEST = Path(
    "reports/training/pillqc_multiclass/"
    "pillqc_multiclass_split_manifest.csv"
)

OUT = Path("reports/training/pillqc_multiclass")
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(MANIFEST)

def image_hash(path):
    h = sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def dhash(path):
    with Image.open(path) as image:
        image = image.convert("L").resize((9, 8))

    pixels = list(image.getdata())
    value = 0
    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            value = (value << 1) | int(left > right)
    return value

records = []

for _, row in df.iterrows():
    path = Path(row["image_path"])
    records.append({
        "path": str(path),
        "split": row["split"],
        "class_name": row["class_name"],
        "sha256": image_hash(path),
        "dhash": dhash(path)
    })

print("Images audited:", len(records))

pairs = []

for a, b in combinations(records, 2):
    if a["split"] == b["split"]:
        continue

    distance = (a["dhash"] ^ b["dhash"]).bit_count()
    exact = a["sha256"] == b["sha256"]

    if distance <= 8 or exact:
        pairs.append({
            "image_a": a["path"],
            "split_a": a["split"],
            "class_a": a["class_name"],
            "image_b": b["path"],
            "split_b": b["split"],
            "class_b": b["class_name"],
            "dhash_distance": distance,
            "exact_duplicate": exact,
            "cross_label": a["class_name"] != b["class_name"]
        })

pairs.sort(
    key=lambda x: (not x["exact_duplicate"], x["dhash_distance"])
)

output = OUT / "cross_split_similarity_audit.csv"

columns = [
    "image_a", "split_a", "class_a",
    "image_b", "split_b", "class_b",
    "dhash_distance", "exact_duplicate", "cross_label"
]

pd.DataFrame(pairs, columns=columns).to_csv(output, index=False)

exact_count = sum(p["exact_duplicate"] for p in pairs)
cross_label_count = sum(p["cross_label"] for p in pairs)

print("\n========== CROSS-SPLIT AUDIT ==========")
print("Cross-split similar pairs (dHash <= 8):", len(pairs))
print("Exact duplicate pairs:", exact_count)
print("Similar pairs crossing labels:", cross_label_count)

print("\nDistance counts:")
if pairs:
    print(
        pd.Series(
            [p["dhash_distance"] for p in pairs]
        ).value_counts().sort_index().to_string()
    )
else:
    print("No similar pairs found.")

print("\nReport:", output)
print("\nNote: dHash similarity is a review signal, not proof of duplication.")
