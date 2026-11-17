import csv
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
IMG_DIR = ROOT / "datasets" / "tablet_detection" / "images" / "all"
LBL_DIR = ROOT / "datasets" / "tablet_detection" / "labels" / "all"
REPORT_DIR = ROOT / "reports"

CSV_OUT = REPORT_DIR / "tablet_dataset_quality.csv"
TXT_OUT = REPORT_DIR / "tablet_dataset_quality.txt"
MONTAGE_OUT = REPORT_DIR / "tablet_dataset_quality_montage.png"

# Existing dataset manifest
def condition(number):
    return "chipped" if 61 <= number <= 80 else "normal"

def parse_label(path):
    boxes = []
    errors = []

    if not path.exists():
        return boxes, ["MISSING_LABEL"]

    lines = path.read_text(encoding="utf-8-sig").strip().splitlines()

    if not lines or not any(line.strip() for line in lines):
        return boxes, ["EMPTY_LABEL"]

    for line_no, line in enumerate(lines, 1):
        parts = line.split()

        if len(parts) != 5:
            errors.append(f"BAD_FIELDS_LINE_{line_no}")
            continue

        try:
            cls = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"NON_NUMERIC_LINE_{line_no}")
            continue

        if cls != 0:
            errors.append(f"UNEXPECTED_CLASS_{cls}")

        if not all(np.isfinite(v) for v in (x, y, w, h)):
            errors.append(f"NONFINITE_LINE_{line_no}")
            continue

        if not (0 <= x <= 1 and 0 <= y <= 1 and
                0 < w <= 1 and 0 < h <= 1):
            errors.append(f"OUT_OF_RANGE_LINE_{line_no}")

        if x-w/2 < 0 or x+w/2 > 1 or y-h/2 < 0 or y+h/2 > 1:
            errors.append(f"BOX_OUTSIDE_IMAGE_LINE_{line_no}")

        boxes.append((cls, x, y, w, h))

    return boxes, errors

def main():
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    images = sorted(
        [p for p in IMG_DIR.iterdir()
         if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}],
        key=lambda p: int(''.join(filter(str.isdigit, p.stem)) or 0)
    )

    rows = []
    montage_items = []

    for image_path in images:
        digits = ''.join(filter(str.isdigit, image_path.stem))
        number = int(digits) if digits else 0
        label_path = LBL_DIR / f"{image_path.stem}.txt"

        frame = cv2.imread(str(image_path))
        boxes, errors = parse_label(label_path)

        if frame is None:
            rows.append({
                "filename": image_path.name,
                "condition": condition(number),
                "image_width": 0,
                "image_height": 0,
                "label_count": len(boxes),
                "label_errors": "UNREADABLE_IMAGE",
                "box_width_pct": "",
                "box_height_pct": "",
                "box_area_pct": "",
                "center_x_pct": "",
                "center_y_pct": "",
                "border_touch": "",
                "quality_flags": "UNREADABLE_IMAGE"
            })
            continue

        height, width = frame.shape[:2]
        flags = list(errors)

        if len(boxes) != 1:
            flags.append(f"BOX_COUNT_{len(boxes)}")

        annotated = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        box_w_pct = box_h_pct = area_pct = center_x_pct = center_y_pct = ""
        border_touch = ""

        for cls, x, y, bw, bh in boxes:
            box_w_pct = round(bw * 100, 3)
            box_h_pct = round(bh * 100, 3)
            area_pct = round(bw * bh * 100, 4)
            center_x_pct = round(x * 100, 3)
            center_y_pct = round(y * 100, 3)

            x1 = int((x-bw/2)*width)
            y1 = int((y-bh/2)*height)
            x2 = int((x+bw/2)*width)
            y2 = int((y+bh/2)*height)

            border_touch = (
                x1 <= 1 or y1 <= 1 or
                x2 >= width-2 or y2 >= height-2
            )

            if bw < 0.05 or bh < 0.05:
                flags.append("VERY_SMALL_BOX")

            if bw * bh < 0.002:
                flags.append("VERY_SMALL_BOX_AREA")

            if border_touch:
                flags.append("BOX_TOUCHES_BORDER")

            cv2.rectangle(
                annotated, (x1,y1), (x2,y2),
                (255, 0, 0), 3
            )

        rows.append({
            "filename": image_path.name,
            "condition": condition(number),
            "image_width": width,
            "image_height": height,
            "label_count": len(boxes),
            "label_errors": ";".join(errors),
            "box_width_pct": box_w_pct,
            "box_height_pct": box_h_pct,
            "box_area_pct": area_pct,
            "center_x_pct": center_x_pct,
            "center_y_pct": center_y_pct,
            "border_touch": border_touch,
            "quality_flags": ";".join(sorted(set(flags))) or "OK"
        })

        montage_items.append((image_path.name, condition(number), annotated, flags))

    fields = list(rows[0].keys()) if rows else []
    with CSV_OUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    total = len(rows)
    valid = sum(r["quality_flags"] == "OK" for r in rows)
    normal = sum(r["condition"] == "normal" for r in rows)
    chipped = sum(r["condition"] == "chipped" for r in rows)
    flagged = [r for r in rows if r["quality_flags"] != "OK"]

    report = f"""
PHARMASIGHT AI - DATASET QUALITY AUDIT
======================================
Images: {total}
Normal: {normal}
Chipped: {chipped}
Unflagged images: {valid}
Flagged images: {len(flagged)}

Flagged filenames:
{chr(10).join(r['filename'] + ': ' + r['quality_flags'] for r in flagged) or 'None'}

Important:
Automated checks cannot confirm whether a tablet is genuinely chipped.
Every flagged box and all chipped labels still require visual inspection.
""".strip()

    TXT_OUT.write_text(report, encoding="utf-8")

    # Contact sheet with annotated YOLO boxes
    cols = 5
    cell_w, cell_h = 300, 250
    canvas = Image.new(
        "RGB",
        (cols*cell_w, ((len(montage_items)+cols-1)//cols)*cell_h),
        "white"
    )
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("arial.ttf", 15)
        small = ImageFont.truetype("arial.ttf", 12)
    except:
        font = small = ImageFont.load_default()

    for i, (name, cond, image, flags) in enumerate(montage_items):
        im = Image.fromarray(image)
        im.thumbnail((cell_w-12, 185))
        x = (i % cols)*cell_w + 6
        y = (i // cols)*cell_h + 5
        canvas.paste(im, (x+(cell_w-12-im.width)//2, y))
        draw.text((x, y+190), name, fill="black", font=font)
        draw.text((x, y+210), f"GT: {cond} | {'FLAGGED' if flags else 'OK'}",
                  fill="black", font=small)

    canvas.save(MONTAGE_OUT)

    print(report)
    print("\nCreated:")
    print(CSV_OUT)
    print(TXT_OUT)
    print(MONTAGE_OUT)

if __name__ == "__main__":
    main()
