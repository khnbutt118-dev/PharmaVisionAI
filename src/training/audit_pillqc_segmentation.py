from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
REPORT = Path("reports/dataset_audit/pillqc")

INPUT = REPORT / "pillqc_center_border_analysis.csv"
OUTPUT = REPORT / "pillqc_segmentation_details.csv"
CONTACT = REPORT / "pillqc_segmentation_contact_sheet.jpg"

df = pd.read_csv(INPUT)

results = []
visuals = {"normal": [], "dirt": [], "chip": []}

for _, record in df.iterrows():
    category = str(record["category"])
    filename = str(record["image"])
    path = ROOT / category / filename

    with Image.open(path) as source:
        image = source.convert("RGB")
        arr = np.asarray(image).astype(np.float32)

    height, width, _ = arr.shape

    # Sample the outer 15% border.
    border = np.zeros((height, width), dtype=bool)
    margin_y = max(1, int(height * 0.15))
    margin_x = max(1, int(width * 0.15))

    border[:margin_y, :] = True
    border[-margin_y:, :] = True
    border[:, :margin_x] = True
    border[:, -margin_x:] = True

    background_rgb = np.median(arr[border], axis=0)

    # Measure colour distance from estimated background.
    distance = np.sqrt(
        np.sum((arr - background_rgb) ** 2, axis=2)
    )

    border_distances = distance[border]

    threshold = max(
        18.0,
        float(np.percentile(border_distances, 95)) + 8.0
    )

    mask = distance > threshold

    foreground_fraction = float(mask.mean())

    ys, xs = np.where(mask)

    if len(xs) > 0:
        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())

        bbox_fraction = (
            ((x_max - x_min + 1) * (y_max - y_min + 1))
            / (width * height)
        )
    else:
        x_min = x_max = y_min = y_max = -1
        bbox_fraction = 0.0

    if foreground_fraction < 0.05:
        status = "possible_detection_failure_small"
    elif foreground_fraction > 0.85:
        status = "possible_detection_failure_large"
    else:
        status = "needs_visual_review"

    results.append({
        "image": filename,
        "category": category,
        "background_r": round(float(background_rgb[0]), 2),
        "background_g": round(float(background_rgb[1]), 2),
        "background_b": round(float(background_rgb[2]), 2),
        "threshold": round(threshold, 2),
        "estimated_tablet_area_fraction": round(
            foreground_fraction, 4
        ),
        "bounding_box_area_fraction": round(
            bbox_fraction, 4
        ),
        "bbox_x_min": x_min,
        "bbox_y_min": y_min,
        "bbox_x_max": x_max,
        "bbox_y_max": y_max,
        "status": status
    })

    # Keep six examples per category for visual inspection.
    if len(visuals[category]) < 6:
        preview = Image.fromarray(arr.astype(np.uint8))
        red_overlay = Image.new("RGB", preview.size, (255, 0, 0))
        blended = Image.blend(preview, red_overlay, 0.40)
        preview.paste(blended, mask=Image.fromarray(
            (mask * 255).astype(np.uint8)
        ))
        visuals[category].append((filename, preview, foreground_fraction))

result = pd.DataFrame(results)
result.to_csv(OUTPUT, index=False)

print("========== SEGMENTATION SUMMARY ==========")

summary = result.groupby("category")[
    "estimated_tablet_area_fraction"
].agg(["count", "mean", "median", "min", "max"])

print(summary.round(4).to_string())

print("\n========== POSSIBLE DETECTION FAILURES ==========")
print(
    result.groupby(["category", "status"])
    .size()
    .to_string()
)

# Create contact sheet: 3 categories, 6 images each.
tile_w, tile_h = 225, 270
canvas = Image.new(
    "RGB",
    (tile_w * 6, tile_h * 3),
    "white"
)
draw = ImageDraw.Draw(canvas)

for row, category in enumerate(["normal", "dirt", "chip"]):
    for col, (filename, preview, fraction) in enumerate(
        visuals[category]
    ):
        preview.thumbnail((205, 205))

        x = col * tile_w
        y = row * tile_h

        canvas.paste(preview, (x + 10, y + 5))
        draw.text(
            (x + 10, y + 215),
            f"{category}: {filename}",
            fill="black"
        )
        draw.text(
            (x + 10, y + 235),
            f"Estimated area: {fraction:.1%}",
            fill="black"
        )

canvas.save(CONTACT, quality=95)

print("\n========== FILES CREATED ==========")
print("Details:", OUTPUT)
print("Contact sheet:", CONTACT)
print("\nOriginal images and manifest were not changed.")
