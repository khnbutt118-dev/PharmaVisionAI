from pathlib import Path
import csv
import math

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(".")
IMAGE_ROOT = ROOT / "datasets/external/pillQC/PillQC/pillQC-main/images"
EXISTING_ROOT = ROOT / "reports/dataset_audit/pillqc/grabcut_masks"

OUT_ROOT = ROOT / "reports/dataset_audit/pillqc/full_method_comparison"
NEW_MASK_ROOT = OUT_ROOT / "centre_seeded_masks"
OVERLAY_ROOT = OUT_ROOT / "centre_seeded_overlays"
SHEET_ROOT = OUT_ROOT / "contact_sheets"

for folder in (NEW_MASK_ROOT, OVERLAY_ROOT, SHEET_ROOT):
    folder.mkdir(parents=True, exist_ok=True)

CATEGORIES = ["normal", "dirt", "chip"]
IMAGE_SIZE = 225
BORDER_FRACTION = 0.15


def centre_seeded_grabcut(image):
    height, width = image.shape[:2]

    # Start with definite background.
    gc_mask = np.full(
        (height, width), cv2.GC_BGD, dtype=np.uint8
    )

    # Outer ellipse: probable tablet foreground.
    outer_axes = (
        max(1, int(width * 0.44)),
        max(1, int(height * 0.44))
    )
    cv2.ellipse(
        gc_mask,
        (width // 2, height // 2),
        outer_axes,
        0, 0, 360,
        cv2.GC_PR_FGD,
        -1
    )

    # Border: definite background.
    bx = max(1, int(width * BORDER_FRACTION))
    by = max(1, int(height * BORDER_FRACTION))
    gc_mask[:by, :] = cv2.GC_BGD
    gc_mask[-by:, :] = cv2.GC_BGD
    gc_mask[:, :bx] = cv2.GC_BGD
    gc_mask[:, -bx:] = cv2.GC_BGD

    # Inner ellipse: definite foreground.
    inner_axes = (
        max(1, int(width * 0.19)),
        max(1, int(height * 0.19))
    )
    cv2.ellipse(
        gc_mask,
        (width // 2, height // 2),
        inner_axes,
        0, 0, 360,
        cv2.GC_FGD,
        -1
    )

    bg_model = np.zeros((1, 65), np.float64)
    fg_model = np.zeros((1, 65), np.float64)

    cv2.grabCut(
        image,
        gc_mask,
        None,
        bg_model,
        fg_model,
        5,
        cv2.GC_INIT_WITH_MASK
    )

    # Keep all foreground components. Do not discard fragments
    # through largest-component filtering.
    return np.where(
        (gc_mask == cv2.GC_FGD) |
        (gc_mask == cv2.GC_PR_FGD),
        255, 0
    ).astype(np.uint8)


def overlay_mask(image, mask):
    result = image.copy()
    red = np.zeros_like(image)
    red[:, :, 2] = 255

    selected = mask > 0
    result[selected] = (
        0.45 * image[selected] + 0.55 * red[selected]
    ).astype(np.uint8)

    return result


def make_sample_panel(image, old_mask, new_mask, label):
    original = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    old_overlay = Image.fromarray(
        cv2.cvtColor(overlay_mask(image, old_mask), cv2.COLOR_BGR2RGB)
    )
    new_overlay = Image.fromarray(
        cv2.cvtColor(overlay_mask(image, new_mask), cv2.COLOR_BGR2RGB)
    )

    thumb = 110
    panel = Image.new("RGB", (thumb * 3 + 20, 155), "white")
    draw = ImageDraw.Draw(panel)

    for index, (picture, caption) in enumerate([
        (original, "Original"),
        (old_overlay, "Existing"),
        (new_overlay, "Centre-seeded")
    ]):
        picture.thumbnail((thumb, thumb))
        x = 5 + index * (thumb + 5)
        panel.paste(picture, (x, 25))
        draw.text((x, 8), caption, fill="black")

    draw.text((5, 138), label, fill="black")
    return panel


def main():
    records = []
    errors = []

    for category in CATEGORIES:
        source_folder = IMAGE_ROOT / category
        old_folder = EXISTING_ROOT / category
        new_folder = NEW_MASK_ROOT / category
        overlay_folder = OVERLAY_ROOT / category

        new_folder.mkdir(parents=True, exist_ok=True)
        overlay_folder.mkdir(parents=True, exist_ok=True)

        image_files = sorted(source_folder.glob("*.jpg"))

        print(f"\nProcessing {category}: {len(image_files)} images")

        for index, image_path in enumerate(image_files, 1):
            try:
                image = cv2.imread(str(image_path))
                if image is None:
                    raise ValueError("Original image could not be read")

                old_path = old_folder / (image_path.stem + ".png")
                old_mask = cv2.imread(
                    str(old_path), cv2.IMREAD_GRAYSCALE
                )
                if old_mask is None:
                    raise ValueError(
                        f"Existing mask missing or unreadable: {old_path}"
                    )

                if old_mask.shape != image.shape[:2]:
                    old_mask = cv2.resize(
                        old_mask,
                        (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_NEAREST
                    )

                new_mask = centre_seeded_grabcut(image)

                new_path = new_folder / (image_path.stem + ".png")
                overlay_path = overlay_folder / (
                    image_path.stem + ".jpg"
                )

                cv2.imwrite(str(new_path), new_mask)
                cv2.imwrite(
                    str(overlay_path),
                    overlay_mask(image, new_mask)
                )

                old_area = float(np.count_nonzero(old_mask)) / old_mask.size
                new_area = float(np.count_nonzero(new_mask)) / new_mask.size

                records.append({
                    "category": category,
                    "image": image_path.name,
                    "existing_area": round(old_area, 5),
                    "centre_seeded_area": round(new_area, 5),
                    "area_change": round(new_area - old_area, 5),
                    "status": "success"
                })

                if index % 25 == 0:
                    print(f"  Completed {index}/{len(image_files)}")

            except Exception as exc:
                errors.append({
                    "category": category,
                    "image": image_path.name,
                    "error": str(exc)
                })

        # Generate contact sheets with 12 samples per page.
        category_records = [
            r for r in records if r["category"] == category
        ]

        panels = []
        for record in category_records:
            image_path = source_folder / record["image"]
            old_path = old_folder / (image_path.stem + ".png")
            new_path = new_folder / (image_path.stem + ".png")

            image = cv2.imread(str(image_path))
            old_mask = cv2.imread(str(old_path), cv2.IMREAD_GRAYSCALE)
            new_mask = cv2.imread(str(new_path), cv2.IMREAD_GRAYSCALE)

            if image is not None and old_mask is not None and new_mask is not None:
                panels.append(
                    make_sample_panel(
                        image,
                        old_mask,
                        new_mask,
                        record["image"]
                    )
                )

        for page_number in range(math.ceil(len(panels) / 12)):
            page_panels = panels[page_number * 12:(page_number + 1) * 12]
            sheet = Image.new("RGB", (400 * 3, 155 * 4), "white")

            for i, panel in enumerate(page_panels):
                x = (i % 3) * 400
                y = (i // 3) * 155
                sheet.paste(panel, (x, y))

            sheet.save(
                SHEET_ROOT /
                f"{category}_page_{page_number + 1:02d}.jpg",
                quality=90
            )

    csv_path = OUT_ROOT / "full_segmentation_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "category", "image", "existing_area",
                "centre_seeded_area", "area_change", "status"
            ]
        )
        writer.writeheader()
        writer.writerows(records)

    error_path = OUT_ROOT / "processing_errors.csv"
    with error_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["category", "image", "error"]
        )
        writer.writeheader()
        writer.writerows(errors)

    print("\n========== FULL SEGMENTATION COMPARISON ==========")
    print("Successfully processed:", len(records))
    print("Errors:", len(errors))
    print("CSV:", csv_path)
    print("New masks:", NEW_MASK_ROOT)
    print("Overlays:", OVERLAY_ROOT)
    print("Contact sheets:", SHEET_ROOT)
    print("\nOriginal images and existing masks were not changed.")


if __name__ == "__main__":
    main()
