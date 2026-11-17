from pathlib import Path
from PIL import Image, ImageOps, ImageDraw
from collections import defaultdict
import hashlib
import csv

root = Path("datasets/external/pillQC/PillQC/pillQC-main")
output = Path("reports/dataset_audit/pillqc")
output.mkdir(parents=True, exist_ok=True)

if not root.exists():
    print("ERROR: Dataset folder not found:", root.resolve())
    raise SystemExit(1)

folders = {
    "normal": root / "images" / "normal",
    "dirt": root / "images" / "dirt",
    "chip": root / "images" / "chip",
    "sourceImages": root / "sourceImages",
}

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
records = []

def difference_hash(image):
    image = image.convert("L").resize((9, 8))
    pixels = list(image.getdata())
    bits = []
    for y in range(8):
        for x in range(8):
            left = pixels[y * 9 + x]
            right = pixels[y * 9 + x + 1]
            bits.append(1 if left > right else 0)
    value = 0
    for bit in bits:
        value = (value << 1) | bit
    return value

def hamming(a, b):
    return (a ^ b).bit_count()

print("========== CONTACT SHEETS AND IMAGE HASHES ==========")

for class_name, folder in folders.items():
    if not folder.exists():
        print("MISSING FOLDER:", folder)
        continue

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )

    print(f"{class_name}: {len(files)} images")

    thumbnails = []
    for path in files:
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                exact_hash = hashlib.sha256(path.read_bytes()).hexdigest()
                dhash = difference_hash(image)

                records.append({
                    "class": class_name,
                    "filename": path.name,
                    "path": str(path),
                    "sha256": exact_hash,
                    "dhash": dhash,
                })

                thumb = ImageOps.contain(image, (140, 115))
                tile = Image.new("RGB", (160, 150), "white")
                x = (160 - thumb.width) // 2
                y = 5 + (115 - thumb.height) // 2
                tile.paste(thumb, (x, y))
                draw = ImageDraw.Draw(tile)
                draw.text((5, 125), path.name[:23], fill="black")
                thumbnails.append(tile)

        except Exception as error:
            print("IMAGE ERROR:", path, error)

    if thumbnails:
        columns = 5
        rows = (len(thumbnails) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * 160, rows * 150), "#dddddd")

        for index, thumb in enumerate(thumbnails):
            x = (index % columns) * 160
            y = (index // columns) * 150
            sheet.paste(thumb, (x, y))

        sheet_path = output / f"{class_name}_contact_sheet.jpg"
        sheet.save(sheet_path, quality=90)
        print("CONTACT SHEET:", sheet_path)

print("\n========== DUPLICATE CANDIDATES ==========")

exact_groups = defaultdict(list)
for record in records:
    exact_groups[record["sha256"]].append(record)

exact_duplicates = [
    group for group in exact_groups.values() if len(group) > 1
]

pairs = []
for i in range(len(records)):
    for j in range(i + 1, len(records)):
        a = records[i]
        b = records[j]

        # Compare only individual pill images here, not source group photos.
        if a["class"] == "sourceImages" or b["class"] == "sourceImages":
            continue

        distance = hamming(a["dhash"], b["dhash"])
        if distance <= 8:
            pairs.append((distance, a, b))

pairs.sort(key=lambda item: item[0])

csv_path = output / "pillqc_duplicate_candidates.csv"
with csv_path.open("w", newline="", encoding="utf-8-sig") as file:
    writer = csv.writer(file)
    writer.writerow([
        "hamming_distance", "class_a", "image_a",
        "class_b", "image_b", "same_class"
    ])
    for distance, a, b in pairs:
        writer.writerow([
            distance, a["class"], a["filename"],
            b["class"], b["filename"],
            a["class"] == b["class"]
        ])

print("Exact duplicate groups:", len(exact_duplicates))
for group in exact_duplicates:
    print("EXACT:", [(r["class"], r["filename"]) for r in group])

print("Near-duplicate candidate pairs (dHash distance <= 8):", len(pairs))
print("Candidate CSV:", csv_path)

print("\n========== AUDIT COMPLETE ==========")
print("Open the contact sheets in:", output.resolve())
