from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import math

root = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
report = Path("reports/dataset_audit/pillqc")
csv_path = report / "pillqc_center_border_analysis.csv"

df = pd.read_csv(csv_path)

categories = ["normal", "dirt", "chip"]
selected = {}

for category in categories:
    group = df[df["category"] == category].sort_values(
        "border_saturation"
    ).reset_index(drop=True)

    indices = [
        round(i * (len(group) - 1) / 11)
        for i in range(12)
    ]

    selected[category] = group.iloc[indices]

tile_w, tile_h = 250, 285
canvas = Image.new("RGB", (tile_w * 6, tile_h * 6 + 60), "white")
draw = ImageDraw.Draw(canvas)

for category_index, category in enumerate(categories):
    start_x = category_index * 2 * tile_w
    draw.text(
        (start_x + 10, 12),
        category.upper() + " — border saturation low to high",
        fill="black"
    )

    for i, (_, record) in enumerate(selected[category].iterrows()):
        col = category_index * 2 + (i % 2)
        row = i // 2
        x = col * tile_w
        y = 60 + row * tile_h

        path = root / category / record["image"]

        with Image.open(path) as original:
            image = original.convert("RGB")
            image.thumbnail((220, 220))

        canvas.paste(image, (x + 10, y + 5))

        label = (
            f'{record["image"]}\n'
            f'Center sat: {record["center_saturation"]:.1f}\n'
            f'Border sat: {record["border_saturation"]:.1f}'
        )

        draw.text((x + 10, y + 230), label, fill="black")

output = report / "pillqc_feature_selected_contact_sheet.jpg"
canvas.save(output, quality=95)

print("========== CONTACT SHEET CREATED ==========")
print("Images selected:", 36)
print("File:", output)
