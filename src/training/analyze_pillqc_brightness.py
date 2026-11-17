import pandas as pd
from pathlib import Path

path = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_image_quality_details.csv"
)

df = pd.read_csv(path)
df = df[df["readable"] == True].copy()

print("========== IMAGE DIMENSIONS ==========")
print("Width values:", sorted(df["width"].unique()))
print("Height values:", sorted(df["height"].unique()))

print("\n========== BRIGHTNESS DISTRIBUTION ==========")

for category in ["normal", "dirt", "chip"]:
    values = df.loc[
        df["category"] == category, "brightness"
    ]

    print(f"\n{category.upper()}")
    print("Count:", len(values))
    print("Minimum:", round(values.min(), 2))
    print("25%:", round(values.quantile(0.25), 2))
    print("Median:", round(values.median(), 2))
    print("75%:", round(values.quantile(0.75), 2))
    print("Maximum:", round(values.max(), 2))
    print("Standard deviation:", round(values.std(), 2))

print("\n========== DARKEST AND BRIGHTEST IMAGES ==========")

print("\nDarkest 10:")
print(
    df.nsmallest(10, "brightness")[
        ["image", "category", "brightness"]
    ].to_string(index=False)
)

print("\nBrightest 10:")
print(
    df.nlargest(10, "brightness")[
        ["image", "category", "brightness"]
    ].to_string(index=False)
)

print("\n========== AUDIT COMPLETE ==========")
