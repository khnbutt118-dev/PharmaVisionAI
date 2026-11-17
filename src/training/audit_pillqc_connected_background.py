from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
REPORT = Path("reports/dataset_audit/pillqc")

df = pd.read_csv(REPORT / "pillqc_center_border_analysis.csv")

def make_border(height, width):
    border = np.zeros((height, width), dtype=bool)

    my = max(1, int(height * 0.15))
    mx = max(1, int(width * 0.15))

    border[:my, :] = True
    border[-my:, :] = True
    border[:, :mx] = True
    border[:, -mx:] = True

    return border

def old_mask(distance, border):
    p95 = np.percentile(distance[border], 95)
    threshold = max(18.0, float(p95) + 8.0)
    return distance > threshold

def connected_background_mask(arr, border, tolerance):
    height, width, _ = arr.shape

    background_colour = np.median(arr[border], axis=0)

    distance = np.sqrt(
        np.sum((arr - background_colour) ** 2, axis=2)
    )

    # Pixels sufficiently similar to the estimated background.
    possible_background = distance <= tolerance

    # Start from all qualifying pixels along the outer border.
    seeds = possible_background & border

    # Follow connected background pixels from the image boundary.
    connected_background = ndimage.binary_propagation(
        seeds,
        structure=np.ones((3, 3), dtype=bool),
        mask=possible_background
    )

    # Everything not connected to the outside is a foreground candidate.
    foreground = ~connected_background

    # Retain the largest connected foreground component.
    labels, count = ndimage.label(
        foreground,
        structure=np.ones((3, 3), dtype=bool)
    )

    if count == 0:
        return foreground

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0

    largest_label = int(np.argmax(sizes))

    return labels == largest_label

def overlay_image(arr, mask):
    base = Image.fromarray(arr.astype(np.uint8))

    red = np.zeros_like(arr, dtype=np.uint8)
    red[:, :, 0] = 255

    red_image = Image.fromarray(red)

    base.paste(
        red_image,
        mask=Image.fromarray((mask * 255).astype(np.uint8))
    )

    return base

results = []
examples = {"normal": [], "dirt": [], "chip": []}

for category in ["normal", "dirt", "chip"]:
    group = df[df["category"] == category].reset_index(drop=True)

    indices = [
        round(i * (len(group) - 1) / 5)
        for i in range(6)
    ]

    examples[category] = [
        str(group.iloc[i]["image"]) for i in indices
    ]

for category in ["normal", "dirt", "chip"]:
    group = df[df["category"] == category]

    for filename in group["image"]:
        path = ROOT / category / str(filename)

        with Image.open(path) as source:
            arr = np.asarray(source.convert("RGB")).astype(np.float32)

        height, width, _ = arr.shape
        border = make_border(height, width)

        background = np.median(arr[border], axis=0)

        distance = np.sqrt(
            np.sum((arr - background) ** 2, axis=2)
        )

        p95 = float(np.percentile(distance[border], 95))

        old = old_mask(distance, border)

        tolerance_1 = max(18.0, p95 + 8.0)
        tolerance_2 = max(24.0, p95 + 16.0)

        new_1 = connected_background_mask(
            arr, border, tolerance_1
        )

        new_2 = connected_background_mask(
            arr, border, tolerance_2
        )

        results.append({
            "category": category,
            "image": str(filename),
            "old_area": round(float(old.mean()), 4),
            "connected_area_1": round(float(new_1.mean()), 4),
            "connected_area_2": round(float(new_2.mean()), 4),
            "tolerance_1": round(tolerance_1, 2),
            "tolerance_2": round(tolerance_2, 2)
        })

result = pd.DataFrame(results)

output = REPORT / "pillqc_connected_background_audit.csv"
result.to_csv(output, index=False)

print("========== SEGMENTATION AREA COMPARISON ==========")

print(
    result.groupby("category")[
        ["old_area", "connected_area_1", "connected_area_2"]
    ].mean().round(4).to_string()
)

# Visual comparison: six examples per category.
columns = 4
tile_w = 225
tile_h = 265

canvas = Image.new(
    "RGB",
    (tile_w * columns, tile_h * 18 + 45),
    "white"
)

draw = ImageDraw.Draw(canvas)

headers = [
    "Original",
    "Previous method",
    "Connected background 1",
    "Connected background 2"
]

for col, header in enumerate(headers):
    draw.text((col * tile_w + 8, 10), header, fill="black")

row_number = 0

for category in ["normal", "dirt", "chip"]:
    for filename in examples[category]:
        path = ROOT / category / filename

        with Image.open(path) as source:
            arr = np.asarray(source.convert("RGB")).astype(np.float32)

        height, width, _ = arr.shape
        border = make_border(height, width)

        background = np.median(arr[border], axis=0)

        distance = np.sqrt(
            np.sum((arr - background) ** 2, axis=2)
        )

        p95 = float(np.percentile(distance[border], 95))

        masks = [
            None,
            old_mask(distance, border),
            connected_background_mask(
                arr, border, max(18.0, p95 + 8.0)
            ),
            connected_background_mask(
                arr, border, max(24.0, p95 + 16.0)
            )
        ]

        y = 45 + row_number * tile_h

        for col, mask in enumerate(masks):
            if mask is None:
                visual = Image.fromarray(arr.astype(np.uint8))
            else:
                visual = overlay_image(arr, mask)

            visual.thumbnail((205, 205))

            canvas.paste(
                visual,
                (col * tile_w + 8, y + 5)
            )

        draw.text(
            (8, y + 215),
            f"{category}: {filename}",
            fill="black"
        )

        row_number += 1

contact = REPORT / "pillqc_connected_background_comparison.jpg"
canvas.save(contact, quality=95)

print("\n========== FILES CREATED ==========")
print("CSV:", output)
print("Contact sheet:", contact)
print("Images processed:", len(result))
print("Original images and manifest unchanged.")
