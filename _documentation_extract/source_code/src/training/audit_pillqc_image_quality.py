from pathlib import Path
from PIL import Image, ImageStat, ImageFilter
import pandas as pd

root = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
output = Path("reports/dataset_audit/pillqc")
output.mkdir(parents=True, exist_ok=True)

records = []

for category in ["normal", "dirt", "chip"]:
    folder = root / category

    if not folder.exists():
        print("Missing category:", folder)
        continue

    for path in sorted(folder.iterdir()):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        try:
            with Image.open(path) as original:
                img = original.convert("RGB")
                gray = img.convert("L")

                stats = ImageStat.Stat(gray)
                edges = gray.filter(ImageFilter.FIND_EDGES)
                edge_stats = ImageStat.Stat(edges)

                records.append({
                    "image": path.name,
                    "category": category,
                    "width": img.width,
                    "height": img.height,
                    "brightness": round(stats.mean[0], 2),
                    "contrast": round(stats.stddev[0], 2),
                    "sharpness_indicator": round(
                        edge_stats.stddev[0], 2
                    ),
                    "file_size_kb": round(
                        path.stat().st_size / 1024, 2
                    ),
                    "readable": True
                })

        except Exception as exc:
            records.append({
                "image": path.name,
                "category": category,
                "readable": False,
                "error": str(exc)
            })

df = pd.DataFrame(records)

detail_file = output / "pillqc_image_quality_details.csv"
summary_file = output / "pillqc_image_quality_summary.csv"

df.to_csv(detail_file, index=False)

valid = df[df["readable"] == True].copy()

summary = valid.groupby("category").agg(
    image_count=("image", "count"),
    average_brightness=("brightness", "mean"),
    average_contrast=("contrast", "mean"),
    average_sharpness=("sharpness_indicator", "mean"),
    average_file_size_kb=("file_size_kb", "mean"),
    unique_dimensions=("width", "nunique")
).round(2)

summary.to_csv(summary_file)

print("\n========== PILLQC IMAGE QUALITY AUDIT ==========")
print("Total images:", len(df))
print("Readable images:", len(valid))
print("Unreadable images:", len(df) - len(valid))

print("\nCATEGORY SUMMARY:")
print(summary.to_string())

print("\nDetailed report:", detail_file)
print("Summary report:", summary_file)
