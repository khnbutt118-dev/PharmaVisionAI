from pathlib import Path
import math

import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path(".")
OUT = Path("reports/training/pillqc_multiclass")

SOURCE = OUT / "cross_split_similarity_audit.csv"
REVIEW = OUT / "closest_cross_split_pairs"
REVIEW.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SOURCE)

close = df[df["dhash_distance"] <= 2].copy()
close = close.sort_values(
    ["dhash_distance", "cross_label"],
    ascending=[True, False]
)

close.to_csv(
    REVIEW / "pairs_distance_0_to_2.csv",
    index=False
)

print("Pairs selected:", len(close))
print("Distance counts:")
print(close["dhash_distance"].value_counts().sort_index())

PAIR_W = 330
PAIR_H = 155
COLS = 2
ROWS = 4
PER_PAGE = COLS * ROWS

def load_thumbnail(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((125, 110))
        return image.copy()

records = close.to_dict("records")
pages = math.ceil(len(records) / PER_PAGE)

for page in range(pages):
    subset = records[
        page * PER_PAGE:(page + 1) * PER_PAGE
    ]

    sheet = Image.new(
        "RGB",
        (COLS * PAIR_W, ROWS * PAIR_H),
        "white"
    )
    draw = ImageDraw.Draw(sheet)

    for index, pair in enumerate(subset):
        x = (index % COLS) * PAIR_W
        y = (index // COLS) * PAIR_H

        image_a = load_thumbnail(pair["image_a"])
        image_b = load_thumbnail(pair["image_b"])

        sheet.paste(image_a, (x + 5, y + 25))
        sheet.paste(image_b, (x + 165, y + 25))

        draw.text(
            (x + 5, y + 5),
            f"Pair {page * PER_PAGE + index + 1} | "
            f"dHash {pair['dhash_distance']}",
            fill="black"
        )

        draw.text(
            (x + 5, y + 138),
            f"A: {pair['class_a']} / {pair['split_a']}",
            fill="black"
        )
        draw.text(
            (x + 165, y + 138),
            f"B: {pair['class_b']} / {pair['split_b']}",
            fill="black"
        )

    sheet.save(
        REVIEW / f"closest_pairs_page_{page + 1:02d}.jpg",
        quality=92
    )

print("\nContact sheets:", REVIEW)
print("Pair details:", REVIEW / "pairs_distance_0_to_2.csv")
