from pathlib import Path
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(".")
df = pd.read_csv("datasets/dataset_manifest.csv")

OUT = Path("reports/dataset_audit/training_contact_sheets")
OUT.mkdir(parents=True, exist_ok=True)

for label in ["normal", "defective"]:

    subset = df[
        (df["split"] == "train") &
        (df["label"] == label)
    ].sample(n=24, random_state=42)

    columns = 4
    rows = 6
    cell_w, cell_h = 260, 260

    sheet = Image.new(
        "RGB",
        (columns * cell_w, rows * cell_h),
        "white"
    )

    draw = ImageDraw.Draw(sheet)

    for i, (_, record) in enumerate(subset.iterrows()):

        path = ROOT / record["image_path"]

        with Image.open(path) as source:
            img = source.convert("RGB")
            img.thumbnail((225, 205))

        x = (i % columns) * cell_w
        y = (i // columns) * cell_h

        image_x = x + (cell_w - img.width) // 2
        sheet.paste(img, (image_x, y + 5))

        draw.text(
            (x + 8, y + 215),
            Path(record["image_path"]).name,
            fill="black"
        )

    output = OUT / f"training_{label}.jpg"
    sheet.save(output, quality=92)

    print("Created:", output)

print("\nTRAINING CONTACT SHEETS COMPLETE")
