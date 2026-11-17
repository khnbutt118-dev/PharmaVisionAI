from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path.cwd()
image_dir = root / "datasets/tablet_detection/images/all"
label_dir = root / "datasets/tablet_detection/labels/all"
output_dir = root / "reports/training/tablet_detection/annotation_preview"

output_dir.mkdir(parents=True, exist_ok=True)

files = sorted(
    [p for p in image_dir.iterdir()
     if p.suffix.lower() in {".jpg", ".jpeg", ".png"}],
    key=lambda p: p.name
)

columns = 6
rows = 6
per_sheet = columns * rows
thumb_w, thumb_h = 240, 190
margin = 12
header_h = 28

for sheet_index in range(math.ceil(len(files) / per_sheet)):
    batch = files[sheet_index * per_sheet:(sheet_index + 1) * per_sheet]

    sheet = Image.new(
        "RGB",
        (columns * (thumb_w + margin) + margin,
         rows * (thumb_h + header_h + margin) + margin),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, image_path in enumerate(batch):
        with Image.open(image_path) as original:
            image = original.convert("RGB")
            width, height = image.size

            label_path = label_dir / f"{image_path.stem}.txt"
            boxes = []

            for line in label_path.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue

                _, cx, cy, bw, bh = parts
                cx, cy, bw, bh = map(float, (cx, cy, bw, bh))

                x1 = (cx - bw / 2) * width
                y1 = (cy - bh / 2) * height
                x2 = (cx + bw / 2) * width
                y2 = (cy + bh / 2) * height
                boxes.append((x1, y1, x2, y2))

            image.thumbnail((thumb_w, thumb_h))
            tile = Image.new("RGB", (thumb_w, thumb_h), "white")

            offset_x = (thumb_w - image.width) // 2
            offset_y = (thumb_h - image.height) // 2
            tile.paste(image, (offset_x, offset_y))

            scale_x = image.width / width
            scale_y = image.height / height
            tile_draw = ImageDraw.Draw(tile)

            for x1, y1, x2, y2 in boxes:
                tile_draw.rectangle(
                    (
                        offset_x + x1 * scale_x,
                        offset_y + y1 * scale_y,
                        offset_x + x2 * scale_x,
                        offset_y + y2 * scale_y
                    ),
                    outline="lime",
                    width=3
                )

        row = index // columns
        col = index % columns
        x = margin + col * (thumb_w + margin)
        y = margin + row * (thumb_h + header_h + margin)

        sheet.paste(tile, (x, y + header_h))
        draw.text(
            (x, y + 5),
            f"{image_path.name} | boxes: {len(boxes)}",
            fill="black"
        )

    output_path = output_dir / f"preview_{sheet_index + 1:02d}.jpg"
    sheet.save(output_path, quality=92)
    print("Created:", output_path)

print("\nTotal images:", len(files))
print("Preview sheets:", math.ceil(len(files) / per_sheet))
