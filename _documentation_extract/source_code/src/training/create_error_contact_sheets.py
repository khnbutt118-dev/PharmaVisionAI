from pathlib import Path
import math
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(".")
PREDICTIONS = ROOT / "reports/evaluation/full_test_set_predictions.csv"
OUTPUT = ROOT / "reports/dataset_audit/contact_sheets"

OUTPUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(PREDICTIONS)

false_negatives = df[
    (df["actual_label"] == "defective") &
    (df["predicted_label"] == "normal")
].sort_values("defective_probability")

false_positives = df[
    (df["actual_label"] == "normal") &
    (df["predicted_label"] == "defective")
].sort_values("defective_probability", ascending=False)

def create_sheets(data, category, columns=4, rows=4):
    per_page = columns * rows
    total_pages = math.ceil(len(data) / per_page)

    for page in range(total_pages):
        subset = data.iloc[
            page * per_page:(page + 1) * per_page
        ]

        cell_w, cell_h = 260, 290
        sheet = Image.new(
            "RGB",
            (columns * cell_w, rows * cell_h),
            "white"
        )
        draw = ImageDraw.Draw(sheet)

        for index, (_, record) in enumerate(subset.iterrows()):
            image_path = ROOT / record["image_path"]

            try:
                with Image.open(image_path) as source:
                    img = source.convert("RGB")
                    img.thumbnail((230, 220))

                x = (index % columns) * cell_w
                y = (index // columns) * cell_h

                image_x = x + (cell_w - img.width) // 2
                sheet.paste(img, (image_x, y + 5))

                filename = Path(record["image_path"]).name
                probability = record["defective_probability"]

                draw.text(
                    (x + 8, y + 232),
                    filename,
                    fill="black"
                )
                draw.text(
                    (x + 8, y + 252),
                    f"Defective probability: {probability:.4f}",
                    fill="black"
                )

            except Exception as error:
                print("Could not open:", image_path, error)

        output_file = OUTPUT / (
            f"{category}_page_{page + 1}.jpg"
        )
        sheet.save(output_file, quality=90)
        print("Created:", output_file)

print("False negatives:", len(false_negatives))
print("False positives:", len(false_positives))

create_sheets(false_negatives, "missed_defective")
create_sheets(false_positives, "false_alarm")

print("\nCONTACT SHEETS COMPLETE")
