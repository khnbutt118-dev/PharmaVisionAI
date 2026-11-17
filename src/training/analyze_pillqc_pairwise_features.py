import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

path = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_center_border_analysis.csv"
)

df = pd.read_csv(path)

features = [
    "center_brightness",
    "border_brightness",
    "center_saturation",
    "border_saturation"
]

comparisons = [
    ("normal", "dirt"),
    ("normal", "chip")
]

results = []

for normal_class, defect_class in comparisons:
    subset = df[
        df["category"].isin([normal_class, defect_class])
    ].copy()

    y = (subset["category"] == defect_class).astype(int)

    for feature in features:
        auc = roc_auc_score(y, subset[feature])

        results.append({
            "comparison": f"{normal_class}_vs_{defect_class}",
            "feature": feature,
            "roc_auc": round(auc, 4),
            "separation_strength": round(max(auc, 1-auc), 4)
        })

result = pd.DataFrame(results)

print("========== PAIRWISE FEATURE AUDIT ==========")
print(result.to_string(index=False))

print("\n========== FEATURE MEDIANS ==========")

print(
    df.groupby("category")[features]
    .median()
    .round(2)
    .to_string()
)

output = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_pairwise_feature_audit.csv"
)

result.to_csv(output, index=False)

print("\nSaved:", output)
print("No images or manifests changed.")
