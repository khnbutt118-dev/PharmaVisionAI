from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
REPORT = Path("reports/dataset_audit/pillqc")

OUTPUT = REPORT / "grabcut_initialization_comparison"
OUTPUT.mkdir(parents=True, exist_ok=True)

TARGETS = [
    ("dirt", "dirt0053.jpg"),
    ("dirt", "dirt0056.jpg"),
    ("chip", "chip0020.jpg"),
    ("chip", "chip0032.jpg"),
    ("chip", "chip0009.jpg"),
    ("chip", "chip0018.jpg"),
    ("normal", "normal0021.jpg"),
    ("normal", "normal0015.jpg"),
]

def make_border(h, w):
    border = np.zeros((h, w), dtype=bool)
    my = max(1, int(h * 0.15))
    mx = max(1, int(w * 0.15))

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


def run_grabcut(arr, grab_mask):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

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


def existing_method(arr):
    h, w, _ = arr.shape
    initial = previous_mask(arr)

    expanded = ndimage.binary_dilation(
        initial,
        iterations=7
    )

    core = ndimage.binary_erosion(
        initial,
        iterations=5
    )

    grab_mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    grab_mask[expanded] = cv2.GC_PR_FGD
    grab_mask[core] = cv2.GC_FGD
    grab_mask[make_border(h, w)] = cv2.GC_BGD

    if not np.any(grab_mask == cv2.GC_FGD):
        cy, cx = h // 2, w // 2
        grab_mask[cy-10:cy+10, cx-10:cx+10] = cv2.GC_FGD

    return run_grabcut(arr, grab_mask)


def centre_seeded_method(arr):
    h, w, _ = arr.shape
    yy, xx = np.ogrid[:h, :w]

    cx = (w - 1) / 2
    cy = (h - 1) / 2

    # Elliptical foreground estimates around the tablet centre.
    outer = (
        ((xx - cx) / (0.44 * w)) ** 2 +
        ((yy - cy) / (0.44 * h)) ** 2
    ) <= 1.0

    inner = (
        ((xx - cx) / (0.19 * w)) ** 2 +
        ((yy - cy) / (0.19 * h)) ** 2
    ) <= 1.0

    grab_mask = np.full(
        (h, w),
        cv2.GC_PR_BGD,
        dtype=np.uint8
    )

    grab_mask[outer] = cv2.GC_PR_FGD
    grab_mask[inner] = cv2.GC_FGD

    # Keep image boundary as definite background.
    grab_mask[make_border(h, w)] = cv2.GC_BGD

    return run_grabcut(arr, grab_mask)


def overlay(arr, mask):
    base = Image.fromarray(arr)
    red = Image.new("RGB", base.size, (255, 0, 0))
    tinted = Image.blend(base, red, 0.40)

    return Image.composite(
        tinted,
        base,
        Image.fromarray(mask.astype(np.uint8) * 255)
    )


rows = []

tile_w = 225
tile_h = 250

canvas = Image.new(
    "RGB",
    (tile_w * 4, 45 + tile_h * len(TARGETS)),
    "white"
)

draw = ImageDraw.Draw(canvas)

headers = [
    "Original",
    "Existing GrabCut",
    "Centre-seeded GrabCut",
    "Centre-seeded mask"
]

for col, header in enumerate(headers):
    draw.text((col * tile_w + 8, 12), header, fill="black")

for index, (category, filename) in enumerate(TARGETS):
    path = ROOT / category / filename

    if not path.exists():
        print("MISSING:", path)
        continue

    with Image.open(path) as source:
        arr = np.asarray(source.convert("RGB")).copy()

    old_mask = existing_method(arr)
    new_mask = centre_seeded_method(arr)

    old_area = float(old_mask.mean())
    new_area = float(new_mask.mean())

    rows.append({
        "category": category,
        "image": filename,
        "existing_area": round(old_area, 4),
        "centre_seeded_area": round(new_area, 4),
        "area_change": round(new_area - old_area, 4)
    })

    mask_path = OUTPUT / f"{category}_{filename}.png"
    Image.fromarray(
        new_mask.astype(np.uint8) * 255
    ).save(mask_path)

    visuals = [
        Image.fromarray(arr),
        overlay(arr, old_mask),
        overlay(arr, new_mask),
        Image.fromarray(
            np.repeat(
                (new_mask.astype(np.uint8) * 255)[:, :, None],
                3,
                axis=2
            )
        )
    ]

    y = 45 + index * tile_h

    for col, visual in enumerate(visuals):
        visual.thumbnail((205, 205))
        canvas.paste(visual, (col * tile_w + 8, y + 22))

    draw.text(
        (8, y + 2),
        f"{category}: {filename} | old {old_area:.1%} | new {new_area:.1%}",
        fill="black"
    )

df = pd.DataFrame(rows)

csv_path = OUTPUT / "initialization_comparison.csv"
sheet_path = OUTPUT / "initialization_comparison.jpg"

df.to_csv(csv_path, index=False)
canvas.save(sheet_path, quality=95)

print("\n========== INITIALIZATION COMPARISON ==========")
print(df.to_string(index=False))
print("\nCSV:", csv_path)
print("Contact sheet:", sheet_path)
print("\nOriginal images and existing masks were not changed.")
