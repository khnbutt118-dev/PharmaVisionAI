from pathlib import Path
from PIL import Image

root = Path(__file__).resolve().parents[2] if False else Path.cwd()

image_dir = root / "datasets" / "tablet_detection" / "images" / "all"
label_dir = root / "datasets" / "tablet_detection" / "labels" / "all"

image_extensions = {".jpg", ".jpeg", ".png"}
images = {
    p.stem: p for p in image_dir.iterdir()
    if p.is_file() and p.suffix.lower() in image_extensions
}
labels = {p.stem: p for p in label_dir.glob("*.txt")}

print("=" * 55)
print("PHARMASIGHT AI - DATASET VALIDATION")
print("=" * 55)
print("Images found:", len(images))
print("Annotation files found:", len(labels))

missing_labels = sorted(set(images) - set(labels))
missing_images = sorted(set(labels) - set(images))
invalid_images = []
invalid_labels = []
total_boxes = 0

for stem, image_path in images.items():
    try:
        with Image.open(image_path) as img:
            img.verify()
    except Exception as exc:
        invalid_images.append((image_path.name, str(exc)))

for stem, label_path in labels.items():
    if stem not in images:
        continue

    try:
        lines = [
            line.strip() for line in label_path.read_text(
                encoding="utf-8"
            ).splitlines() if line.strip()
        ]

        if not lines:
            invalid_labels.append((label_path.name, "Empty annotation"))
            continue

        for line_number, line in enumerate(lines, start=1):
            parts = line.split()

            if len(parts) != 5:
                raise ValueError(
                    f"Line {line_number}: expected 5 values"
                )

            class_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])

            if class_id != 0:
                raise ValueError(
                    f"Line {line_number}: expected class 0"
                )

            if not all(0 <= v <= 1 for v in (x, y, w, h)):
                raise ValueError(
                    f"Line {line_number}: values outside 0–1"
                )

            if w <= 0 or h <= 0:
                raise ValueError(
                    f"Line {line_number}: zero or negative box size"
                )

            if x - w / 2 < 0 or y - h / 2 < 0:
                raise ValueError(
                    f"Line {line_number}: box starts outside image"
                )

            if x + w / 2 > 1 or y + h / 2 > 1:
                raise ValueError(
                    f"Line {line_number}: box ends outside image"
                )

            total_boxes += 1

    except Exception as exc:
        invalid_labels.append((label_path.name, str(exc)))

print("\nMissing annotation files:", len(missing_labels))
print("Missing image files:", len(missing_images))
print("Unreadable images:", len(invalid_images))
print("Invalid annotation files:", len(invalid_labels))
print("Total tablet boxes:", total_boxes)

if missing_labels:
    print("\nImages without annotations:", missing_labels[:10])

if missing_images:
    print("\nAnnotations without images:", missing_images[:10])

if invalid_images:
    print("\nUnreadable image examples:", invalid_images[:5])

if invalid_labels:
    print("\nInvalid annotation examples:", invalid_labels[:10])

if (
    len(images) == len(labels)
    and not missing_labels
    and not missing_images
    and not invalid_images
    and not invalid_labels
):
    print("\nRESULT: STRUCTURAL VALIDATION PASSED")
else:
    print("\nRESULT: ISSUES FOUND - DO NOT TRAIN YET")
