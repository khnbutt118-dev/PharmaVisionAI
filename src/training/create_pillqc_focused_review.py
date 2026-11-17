from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

csv_path = Path("reports/dataset_audit/pillqc/pillqc_grabcut_full_audit.csv")
output_dir = Path("reports/dataset_audit/pillqc/focused_review")

output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
df = df[df["status"] == "success"].copy()

for category in ["normal", "dirt", "chip"]:
    part = df[df["category"] == category].copy()

    lowest = part.nsmallest(8, "grabcut_area")
    highest = part.nlargest(4, "grabcut_area")

    selected = pd.concat([lowest, highest]).drop_duplicates("image")

    print(f"\n{category.upper()} selected images:")
    print(selected["image"].tolist())

    records = selected.to_dict("records")

    for page_number, start in enumerate(range(0, len(records), 6), start=1):
        page = records[start:start + 6]

        tile_w = 225
        tile_h = 205

        canvas = Image.new(
            "RGB",
            (tile_w * 3, 45 + tile_h * len(page)),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        for col, title in enumerate(
            ["ORIGINAL", "GRABCUT OVERLAY", "BINARY MASK"]
        ):
            draw.text((col * tile_w + 8, 12), title, fill="black")

        for row, record in enumerate(page):
            y = 45 + row * tile_h

            draw.text(
                (8, y + 2),
                f'{record["image"]} | Coverage: {float(record["grabcut_area"]):.2%}',
                fill="black"
            )

            images = [
                Image.open(record["image_path"]).convert("RGB")
                if "image_path" in record and Path(str(record["image_path"])).exists()
                else None,
                Image.open(record["overlay_path"]).convert("RGB"),
                Image.open(record["mask_path"]).convert("RGB")
            ]

            # Original path is reconstructed from the known dataset structure.
            if images[0] is None:
                original_path = (
                    Path("datasets/external/pillQC/PillQC/pillQC-main/images")
                    / category / record["image"]
                )
                images[0] = Image.open(original_path).convert("RGB")

            for col, image in enumerate(images):
                image.thumbnail((195, 165))
                x = col * tile_w + 8
                canvas.paste(image, (x, y + 25))

        output = output_dir / f"{category}_review_page_{page_number}.jpg"
        canvas.save(output, quality=95)
        print("Saved:", output)

print("\nFocused review sheets created:", output_dir)
