from pathlib import Path
import numpy as np
import pandas as pd
import cv2
from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
REPORT = Path("reports/dataset_audit/pillqc")

targets = [
    ("dirt", "dirt0056.jpg"),
    ("dirt", "dirt0083.jpg"),
    ("chip", "chip0009.jpg"),
    ("chip", "chip0018.jpg"),
    ("chip", "chip0035.jpg"),
]

def make_border(h, w):
    border = np.zeros((h, w), dtype=bool)
    my, mx = int(h * 0.15), int(w * 0.15)
    border[:my, :] = True
    border[-my:, :] = True
    border[:, :mx] = True
    border[:, -mx:] = True
    return border

def previous_mask(arr):
    h, w, _ = arr.shape
    border = make_border(h, w)
    background = np.median(arr[border], axis=0)

    distance = np.sqrt(
        np.sum((arr - background) ** 2, axis=2)
    )

    p95 = float(np.percentile(distance[border], 95))
    tolerance = max(24.0, p95 + 16.0)

    possible_bg = distance <= tolerance
    seeds = possible_bg & border

    connected_bg = ndimage.binary_propagation(
        seeds,
        structure=np.ones((3, 3), dtype=bool),
        mask=possible_bg
    )

    foreground = ~connected_bg

    labels, count = ndimage.label(
        foreground,
        structure=np.ones((3, 3), dtype=bool)
    )

    if count == 0:
        return foreground

    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    return labels == int(np.argmax(sizes))

def grabcut_refine(arr, initial):
    h, w, _ = arr.shape

    # OpenCV requires an 8-bit BGR image.
    bgr = cv2.cvtColor(
        arr.astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

    # Expand the initial tablet estimate slightly.
    expanded = ndimage.binary_dilation(
        initial,
        iterations=7
    )

    core = ndimage.binary_erosion(
        initial,
        iterations=5
    )

    # GrabCut labels:
    # 0 = definite background
    # 1 = definite foreground
    # 2 = probable background
    # 3 = probable foreground
    grab_mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    grab_mask[expanded] = cv2.GC_PR_FGD
    grab_mask[core] = cv2.GC_FGD

    # Keep the outer border as definite background.
    border = make_border(h, w)
    grab_mask[border] = cv2.GC_BGD

    # If erosion removed all definite foreground seeds,
    # use a small central region as a fallback.
    if not np.any(grab_mask == cv2.GC_FGD):
        cy, cx = h // 2, w // 2
        grab_mask[
            cy - 10:cy + 10,
            cx - 10:cx + 10
        ] = cv2.GC_FGD

    bg_model = np.zeros((1, 65), dtype=np.float64)
    fg_model = np.zeros((1, 65), dtype=np.float64)

    cv2.grabCut(
        bgr,
        grab_mask,
        None,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    return np.isin(
        grab_mask,
        [cv2.GC_FGD, cv2.GC_PR_FGD]
    )

def overlay(arr, mask):
    base = Image.fromarray(arr.astype(np.uint8))
    red = Image.new("RGB", base.size, (255, 0, 0))
    base.paste(
        red,
        mask=Image.fromarray(
            (mask.astype(np.uint8) * 255)
        )
    )
    return base

rows = []
tile_w, tile_h = 225, 255

canvas = Image.new(
    "RGB",
    (tile_w * 3, tile_h * len(targets) + 40),
    "white"
)

draw = ImageDraw.Draw(canvas)

headers = ["Original", "Previous method", "GrabCut"]

for col, header in enumerate(headers):
    draw.text(
        (col * tile_w + 8, 10),
        header,
        fill="black"
    )

for row, (category, filename) in enumerate(targets):
    path = ROOT / category / filename

    if not path.exists():
        print("MISSING:", path)
        continue

    with Image.open(path) as source:
        arr = np.asarray(
            source.convert("RGB")
        ).copy()

    initial = previous_mask(arr)
    refined = grabcut_refine(arr, initial)

    rows.append({
        "category": category,
        "image": filename,
        "previous_area": round(float(initial.mean()), 4),
        "grabcut_area": round(float(refined.mean()), 4),
        "area_change": round(
            float(refined.mean() - initial.mean()), 4
        )
    })

    visuals = [
        Image.fromarray(arr),
        overlay(arr, initial),
        overlay(arr, refined)
    ]

    y = 40 + row * tile_h

    for col, visual in enumerate(visuals):
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

result = pd.DataFrame(rows)

csv_path = REPORT / "pillqc_grabcut_test_results.csv"
image_path = REPORT / "pillqc_grabcut_test_contact_sheet.jpg"

result.to_csv(csv_path, index=False)
canvas.save(image_path, quality=95)

print("========== GRABCUT TEST RESULTS ==========")
print(result.to_string(index=False))

print("\nSaved:", csv_path)
print("Contact sheet:", image_path)
print("\nOnly five images were tested.")
print("Original images and manifest were not changed.")
