from pathlib import Path
from PIL import Image, ImageDraw
import pandas as pd
import math

project = Path(".").resolve()
csv_path = project / "reports/dataset_audit/pillqc/pillqc_duplicate_candidates.csv"
image_root = project / "datasets/external/pillQC/PillQC/pillQC-main/images"

# Find the actual image folder if the nesting differs.
if not image_root.exists():
    matches = list((project / "datasets/external").rglob("normal"))
    matches = [
        p.parent for p in matches
        if p.is_dir() and p.parent.name.lower() == "images"
        and (p.parent / "dirt").is_dir()
        and (p.parent / "chip").is_dir()
    ]
    if len(matches) != 1:
        print("ERROR: Could not uniquely locate the PillQC images folder.")
        raise SystemExit(1)
    image_root = matches[0]

df = pd.read_csv(csv_path)
output = project / "reports/dataset_audit/pillqc"
output.mkdir(parents=True, exist_ok=True)

def get_image(category, filename):
    path = image_root / str(category).lower() / str(filename)
    if not path.exists():
        return None
    try:
        return Image.open(path).convert("RGB")
    except Exception:
        return None

def create_sheet(data, filename, title):
    if len(data) == 0:
        print("No pairs to display:", filename)
        return

    # Limit sheet size for easy inspection.
    data = data.head(60).reset_index(drop=True)

    tile_w, tile_h = 520, 250
    columns = 2
    rows = math.ceil(len(data) / columns)
    sheet = Image.new(
        "RGB",
        (columns * tile_w, 45 + rows * tile_h),
        "white"
    )
    draw = ImageDraw.Draw(sheet)
    draw.text((15, 12), title, fill="black")

    for i, row in data.iterrows():
        x = (i % columns) * tile_w
        y = 45 + (i // columns) * tile_h

        draw.rectangle(
            [x + 2, y + 2, x + tile_w - 4, y + tile_h - 4],
            outline="gray"
        )

        left = get_image(row["class_a"], row["image_a"])
        right = get_image(row["class_b"], row["image_b"])

        if left:
            left.thumbnail((235, 195))
            sheet.paste(left, (x + 8, y + 35))
        if right:
            right.thumbnail((235, 195))
            sheet.paste(right, (x + 270, y + 35))

        caption = (
            f'{row["class_a"]}/{row["image_a"]}  <->  '
            f'{row["class_b"]}/{row["image_b"]}'
        )
        draw.text((x + 8, y + 8), caption[:75], fill="black")
        draw.text(
            (x + 8, y + 220),
            f'Hamming distance: {row["hamming_distance"]}',
            fill="blue"
        )

    path = output / filename
    sheet.save(path, quality=90)
    print("CONTACT SHEET:", path)

cross = df[
    df["class_a"].str.lower() != df["class_b"].str.lower()
].sort_values(["hamming_distance", "image_a", "image_b"])

same = df[
    df["class_a"].str.lower() == df["class_b"].str.lower()
].sort_values(["hamming_distance", "image_a", "image_b"])

print("Cross-category pairs with distance <= 2:",
      len(cross[cross["hamming_distance"] <= 2]))
print("Same-category pairs with distance <= 1:",
      len(same[same["hamming_distance"] <= 1]))

create_sheet(
    cross[cross["hamming_distance"] <= 2],
    "cross_category_strong_matches.jpg",
    "CROSS-CATEGORY STRONG MATCHES"
)

create_sheet(
    same[same["hamming_distance"] <= 1],
    "same_category_strong_matches.jpg",
    "SAME-CATEGORY STRONG MATCHES"
)

print("\nPAIR COMPARISON SHEETS COMPLETE")
