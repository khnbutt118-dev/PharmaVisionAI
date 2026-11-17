from pathlib import Path
import pandas as pd

path = Path("reports/dataset_audit/pillqc/pillqc_duplicate_candidates.csv")
df = pd.read_csv(path)

print("\n========== PAIR COUNTS BY CATEGORY ==========")

df["pair_category"] = (
    df["class_a"].astype(str).str.lower()
    + " <-> "
    + df["class_b"].astype(str).str.lower()
)

print(df["pair_category"].value_counts().to_string())

print("\n========== SAME CLASS CHECK ==========")
print(df["same_class"].value_counts(dropna=False).to_string())

print("\n========== CROSS-CATEGORY PAIRS ==========")
cross = df[
    df["class_a"].astype(str).str.lower()
    != df["class_b"].astype(str).str.lower()
]

print("Cross-category candidate pairs:", len(cross))

if len(cross):
    print(cross.head(50).to_string(index=False))
else:
    print("No cross-category candidate pairs found.")

print("\n========== EXACT VISUAL MATCHES ==========")
exact = df[df["hamming_distance"] == 0]
print("Distance 0 pairs:", len(exact))
print(exact.head(30).to_string(index=False))

print("\n========== DISTANCE BY CATEGORY ==========")
print(
    df.groupby("pair_category")["hamming_distance"]
    .agg(["count", "min", "mean", "max"])
    .to_string()
)

print("\n========== CORRECTED AUDIT COMPLETE ==========")
