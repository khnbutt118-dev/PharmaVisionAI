from pathlib import Path
from PIL import Image
import numpy as np
import pandas as pd

root = Path("datasets/external/pillQC/PillQC/pillQC-main/images")
output = Path("reports/dataset_audit/pillqc")
output.mkdir(parents=True, exist_ok=True)

records = []

for category in ["normal", "dirt", "chip"]:
    for path in sorted((root / category).glob("*")):
        if path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
            continue

        with Image.open(path) as original:
            arr = np.asarray(original.convert("RGB"), dtype=np.float32)

        h, w, _ = arr.shape

        # Central 50% of the image
        center = arr[int(h*.25):int(h*.75),
                     int(w*.25):int(w*.75)]

        # Outer 15% border
        mask = np.ones((h, w), dtype=bool)
        mask[int(h*.15):int(h*.85),
             int(w*.15):int(w*.85)] = False
        border = arr[mask]

        def measurements(pixels):
            luminance = (
                0.299 * pixels[:, 0]
                + 0.587 * pixels[:, 1]
                + 0.114 * pixels[:, 2]
            )

            saturation = (
                pixels.max(axis=1) - pixels.min(axis=1)
            )

            return (
                float(luminance.mean()),
                float(saturation.mean())
            )

        center_brightness, center_saturation = measurements(
            center.reshape(-1, 3)
        )

        border_brightness, border_saturation = measurements(
            border.reshape(-1, 3)
        )

        records.append({
            "image": path.name,
            "category": category,
            "center_brightness": round(center_brightness, 2),
            "border_brightness": round(border_brightness, 2),
            "center_saturation": round(center_saturation, 2),
            "border_saturation": round(border_saturation, 2)
        })

df = pd.DataFrame(records)

detail = output / "pillqc_center_border_analysis.csv"
summary = output / "pillqc_center_border_summary.csv"

df.to_csv(detail, index=False)

result = df.groupby("category").agg(
    image_count=("image", "count"),
    center_brightness=("center_brightness", "mean"),
    border_brightness=("border_brightness", "mean"),
    center_saturation=("center_saturation", "mean"),
    border_saturation=("border_saturation", "mean")
).round(2)

result.to_csv(summary)

print("========== CENTRE VS BORDER ANALYSIS ==========")
print(result.to_string())

print("\nDetailed report:", detail)
print("Summary report:", summary)
