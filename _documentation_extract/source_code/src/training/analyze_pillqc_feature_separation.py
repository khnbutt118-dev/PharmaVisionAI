import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

path = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_center_border_analysis.csv"
)

df = pd.read_csv(path)

# Normal = 0, dirt/chip = 1
df["is_defective"] = (
    df["category"].isin(["dirt", "chip"])
).astype(int)

features = [
    "center_brightness",
    "border_brightness",
    "center_saturation",
    "border_saturation"
]

print("========== FEATURE SEPARATION AUDIT ==========")
print("Normal images:", int((df["is_defective"] == 0).sum()))
print("Defective images:", int((df["is_defective"] == 1).sum()))

results = []

for feature in features:
    auc = roc_auc_score(
        df["is_defective"],
        df[feature]
    )

    results.append({
        "feature": feature,
        "roc_auc": round(auc, 4),
        "separation_strength": round(max(auc, 1 - auc), 4)
    })

result = pd.DataFrame(results).sort_values(
    "separation_strength", ascending=False
)

print("\nFEATURE RESULTS:")
print(result.to_string(index=False))

print("\n========== CLASS MEDIANS ==========")

print(
    df.groupby("category")[features]
    .median()
    .round(2)
    .to_string()
)

output = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_feature_separation.csv"
)

result.to_csv(output, index=False)

print("\nSaved:", output)
print("\nAudit complete. No images or manifests changed.")
