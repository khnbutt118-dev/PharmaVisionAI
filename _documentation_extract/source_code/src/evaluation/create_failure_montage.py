import csv, math
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = ROOT / "reports" / "tablet_webcam_baseline.csv"
IMAGE_DIR = ROOT / "datasets" / "tablet_detection" / "images" / "all"
OUT = ROOT / "reports" / "tablet_failure_montage.png"

with CSV_PATH.open(encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))

failures = [
    r for r in rows
    if r["prediction"] == "review"
    or r["ground_truth"] != r["prediction"]
]

cols = 4
cell_w, cell_h = 360, 300
pad = 12
canvas = Image.new(
    "RGB",
    (cols * cell_w, math.ceil(len(failures)/cols) * cell_h),
    "white"
)
draw = ImageDraw.Draw(canvas)

try:
    font = ImageFont.truetype("arial.ttf", 17)
    small = ImageFont.truetype("arial.ttf", 14)
except:
    font = small = ImageFont.load_default()

for i, r in enumerate(failures):
    image_path = IMAGE_DIR / r["filename"]
    im = Image.open(image_path).convert("RGB")
    im.thumbnail((cell_w - 2*pad, 205))

    x = (i % cols) * cell_w + pad
    y = (i // cols) * cell_h + pad
    canvas.paste(im, (x + (cell_w-2*pad-im.width)//2, y))

    prob = float(r["p_defective"])
    if r["prediction"] == "review":
        category = "REVIEW"
    elif r["ground_truth"] == "normal":
        category = "FALSE POSITIVE"
    else:
        category = "FALSE NEGATIVE"

    ty = y + 212
    draw.text((x, ty), r["filename"], fill="black", font=font)
    draw.text(
        (x, ty+24),
        f"GT: {r['ground_truth']} | Pred: {r['prediction']}",
        fill="black", font=small
    )
    draw.text(
        (x, ty+45),
        f"P(defective): {prob:.3f} | {category}",
        fill="black", font=small
    )

canvas.save(OUT)
print(f"Montage created: {OUT}")
print(f"Images included: {len(failures)}")
