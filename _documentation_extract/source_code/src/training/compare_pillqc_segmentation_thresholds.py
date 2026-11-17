from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

ROOT = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
REPORT = Path("reports/dataset_audit/pillqc")

df = pd.read_csv(REPORT / "pillqc_center_border_analysis.csv")

results = []
examples = {"normal": [], "dirt": [], "chip": []}

for category in ["normal", "dirt", "chip"]:
    group = df[df["category"] == category].reset_index(drop=True)

    indices = [
        round(i * (len(group) - 1) / 5)
        for i in range(6)
    ]

    for index in indices:
        record = group.iloc[index]
        examples[category].append(str(record["image"]))

for category in ["normal", "dirt", "chip"]:
    for filename in df[df["category"] == category]["image"]:
        path = ROOT / category / str(filename)

        with Image.open(path) as source:
            arr = np.asarray(source.convert("RGB")).astype(np.float32)

        h, w, _ = arr.shape

        border = np.zeros((h, w), dtype=bool)
        my = max(1, int(h * 0.15))
        mx = max(1, int(w * 0.15))

        border[:my, :] = True
        border[-my:, :] = True
        border[:, :mx] = True
        border[:, -mx:] = True

        background = np.median(arr[border], axis=0)

        distance = np.sqrt(
            np.sum((arr - background) ** 2, axis=2)
        )

        p95 = float(np.percentile(distance[border], 95))

        thresholds = {
            "lower": max(10.0, p95),
            "current": max(18.0, p95 + 8.0),
            "higher": max(22.0, p95 + 12.0)
        }

        row = {
            "category": category,
            "image": str(filename),
            "border_p95_distance": round(p95, 2)
        }

        for name, threshold in thresholds.items():
            mask = distance > threshold
            row[f"{name}_threshold"] = round(threshold, 2)
            row[f"{name}_area_fraction"] = round(
                float(mask.mean()), 4
            )

        results.append(row)

result = pd.DataFrame(results)

output_csv = REPORT / "pillqc_threshold_comparison.csv"
result.to_csv(output_csv, index=False)

print("========== THRESHOLD COMPARISON ==========")

print(
    result.groupby("category")[
        [
            "lower_area_fraction",
            "current_area_fraction",
            "higher_area_fraction"
        ]
    ].mean().round(4).to_string()
)

# Build visual comparison: original, lower, current, higher.
tile_w = 225
tile_h = 265
canvas = Image.new(
    "RGB",
    (tile_w * 4, tile_h * 18 + 35),
    "white"
)
draw = ImageDraw.Draw(canvas)

headers = ["Original", "Lower threshold", "Current", "Higher threshold"]

for col, header in enumerate(headers):
    draw.text((col * tile_w + 10, 10), header, fill="black")

row_number = 0

for category in ["normal", "dirt", "chip"]:
    for filename in examples[category]:
        path = ROOT / category / filename

        with Image.open(path) as source:
            arr = np.asarray(source.convert("RGB")).astype(np.float32)

        h, w, _ = arr.shape

        border = np.zeros((h, w), dtype=bool)
        my = max(1, int(h * 0.15))
        mx = max(1, int(w * 0.15))

        border[:my, :] = True
        border[-my:, :] = True
        border[:, :mx] = True
        border[:, -mx:] = True

        background = np.median(arr[border], axis=0)
        distance = np.sqrt(
            np.sum((arr - background) ** 2, axis=2)
        )

        p95 = float(np.percentile(distance[border], 95))

        thresholds = [
            None,
            max(10.0, p95),
            max(18.0, p95 + 8.0),
            max(22.0, p95 + 12.0)
        ]

        original = Image.fromarray(arr.astype(np.uint8))
        original.thumbnail((205, 205))

        y = 35 + row_number * tile_h

        canvas.paste(original, (10, y + 5))

        for col in range(1, 4):
            mask = distance > thresholds[col]

            overlay = np.clip(
                arr * 0.60 + np.array([255, 0, 0]) * 0.40,
                0, 255
            ).astype(np.uint8)

            visual = Image.fromarray(arr.astype(np.uint8))
            visual.paste(
                Image.fromarray(overlay),
                mask=Image.fromarray((mask * 255).astype(np.uint8))
            )

            visual.thumbnail((205, 205))
            canvas.paste(
                visual,
                (col * tile_w + 10, y + 5)
            )

        draw.text(
            (10, y + 215),
            f"{category}: {filename}",
            fill="black"
        )

        row_number += 1

contact = REPORT / "pillqc_threshold_comparison.jpg"
canvas.save(contact, quality=95)

print("\n========== FILES CREATED ==========")
print("CSV:", output_csv)
print("Contact sheet:", contact)
print("All 330 images compared.")
print("Original images and manifest unchanged.")
