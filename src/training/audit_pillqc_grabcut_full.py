from pathlib import Path
import numpy as np
import pandas as pd
import cv2

from PIL import Image, ImageDraw
from scipy import ndimage

ROOT = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
REPORT = Path("reports/dataset_audit/pillqc")

MASK_DIR = REPORT / "grabcut_masks"
SHEET_DIR = REPORT / "grabcut_contact_sheets"

CATEGORIES = ["normal", "dirt", "chip"]

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

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


def grabcut_refine(arr, initial):
    h, w, _ = arr.shape

    bgr = cv2.cvtColor(
        arr.astype(np.uint8),
        cv2.COLOR_RGB2BGR
    )

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

    border = make_border(h, w)
    grab_mask[border] = cv2.GC_BGD

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


def make_overlay(arr, mask):
    base = Image.fromarray(arr)
    red = Image.new("RGB", base.size, (255, 0, 0))

    tinted = Image.blend(base, red, 0.40)

    return Image.composite(
        tinted,
        base,
        Image.fromarray(mask.astype(np.uint8) * 255)
    )


def save_contact_sheets(category, records):
    page_size = 18
    tile_w = 140
    tile_h = 155
    columns = 3
    rows_per_page = 6

    category_dir = SHEET_DIR / category
    category_dir.mkdir(parents=True, exist_ok=True)

    for page_number, start in enumerate(
        range(0, len(records), page_size), start=1
    ):
        page_records = records[start:start + page_size]

        canvas = Image.new(
            "RGB",
            (columns * tile_w, 40 + rows_per_page * tile_h),
            "white"
        )

        draw = ImageDraw.Draw(canvas)

        draw.text(
            (8, 10),
            f"{category.upper()} - Page {page_number} | Original / GrabCut / Mask",
            fill="black"
        )

        for index, record in enumerate(page_records):
            row = index // columns
            col = index % columns

            x = col * tile_w
            y = 40 + row * tile_h

            original = Image.open(record["original_path"]).convert("RGB")
            overlay = Image.open(record["overlay_path"]).convert("RGB")
            mask = Image.open(record["mask_path"]).convert("RGB")

            for offset, image in zip(
                [0, 40, 80],
                [original, overlay, mask]
            ):
                image.thumbnail((38, 115))
                canvas.paste(image, (x + offset + 2, y + 20))

            draw.text(
                (x + 2, y + 2),
                record["image"][:19],
                fill="black"
            )

            draw.text(
                (x + 2, y + 138),
                f'Coverage: {record["grabcut_area"]:.1%}',
                fill="black"
            )

        output = category_dir / f"{category}_page_{page_number:02d}.jpg"
        canvas.save(output, quality=92)

        print("Contact sheet saved:", output)


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    SHEET_DIR.mkdir(parents=True, exist_ok=True)

    results = []
    visual_records = []

    for category in CATEGORIES:
        category_path = ROOT / category

        if not category_path.exists():
            print("MISSING CATEGORY:", category_path)
            continue

        output_mask_dir = MASK_DIR / category
        output_overlay_dir = REPORT / "grabcut_overlays" / category

        output_mask_dir.mkdir(parents=True, exist_ok=True)
        output_overlay_dir.mkdir(parents=True, exist_ok=True)

        files = sorted(
            p for p in category_path.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )

        print(f"\n========== {category.upper()} ==========")
        print("Images found:", len(files))

        for number, path in enumerate(files, start=1):
            try:
                with Image.open(path) as source:
                    arr = np.asarray(
                        source.convert("RGB")
                    ).copy()

                initial = previous_mask(arr)
                refined = grabcut_refine(arr, initial)

                mask_image = Image.fromarray(
                    refined.astype(np.uint8) * 255
                )

                overlay_image = make_overlay(arr, refined)

                mask_path = output_mask_dir / f"{path.stem}.png"
                overlay_path = output_overlay_dir / f"{path.stem}.jpg"

                mask_image.save(mask_path)
                overlay_image.save(overlay_path, quality=95)

                initial_area = float(initial.mean())
                final_area = float(refined.mean())

                record = {
                    "category": category,
                    "image": path.name,
                    "width": arr.shape[1],
                    "height": arr.shape[0],
                    "previous_area": round(initial_area, 4),
                    "grabcut_area": round(final_area, 4),
                    "area_change": round(final_area - initial_area, 4),
                    "status": "success",
                    "error": "",
                    "mask_path": str(mask_path),
                    "overlay_path": str(overlay_path)
                }

                results.append(record)

                visual_records.append({
                    "image": path.name,
                    "original_path": path,
                    "mask_path": mask_path,
                    "overlay_path": overlay_path,
                    "grabcut_area": final_area
                })

                if number % 25 == 0 or number == len(files):
                    print(f"Processed {number}/{len(files)}")

            except Exception as exc:
                results.append({
                    "category": category,
                    "image": path.name,
                    "width": None,
                    "height": None,
                    "previous_area": None,
                    "grabcut_area": None,
                    "area_change": None,
                    "status": "error",
                    "error": str(exc),
                    "mask_path": "",
                    "overlay_path": ""
                })

                print("ERROR:", path.name, str(exc))

        category_records = [
            r for r in visual_records
            if r["original_path"].parent.name == category
        ]

        save_contact_sheets(category, category_records)

    df = pd.DataFrame(results)

    csv_path = REPORT / "pillqc_grabcut_full_audit.csv"
    df.to_csv(csv_path, index=False)

    print("\n========== FULL GRABCUT AUDIT SUMMARY ==========")

    if not df.empty:
        print("\nStatus counts:")
        print(df.groupby(["category", "status"]).size().to_string())

        print("\nSuccessful image coverage:")
        successful = df[df["status"] == "success"]

        if not successful.empty:
            print(
                successful.groupby("category")["grabcut_area"]
                .agg(["count", "mean", "median", "min", "max"])
                .round(4)
                .to_string()
            )

        print("\nFailed images:")
        failures = df[df["status"] == "error"]

        if failures.empty:
            print("None")
        else:
            print(failures[["category", "image", "error"]].to_string(index=False))

    print("\nCSV report:", csv_path)
    print("Masks:", MASK_DIR)
    print("Overlays:", REPORT / "grabcut_overlays")
    print("Contact sheets:", SHEET_DIR)
    print("\nOriginal images and manifests were not changed.")


if __name__ == "__main__":
    main()
