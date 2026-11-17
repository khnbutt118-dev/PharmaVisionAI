import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

path = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_center_border_analysis.csv"
)

df = pd.read_csv(path)

feature_sets = {
    "Background only": [
        "border_brightness",
        "border_saturation"
    ],
    "Centre only": [
        "center_brightness",
        "center_saturation"
    ],
    "All four features": [
        "center_brightness",
        "center_saturation",
        "border_brightness",
        "border_saturation"
    ]
}

X_labels = df["category"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

results = []

for name, features in feature_sets.items():
    X = df[features]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000)
    )

    scores = cross_validate(
        model,
        X,
        X_labels,
        cv=cv,
        scoring=["accuracy", "balanced_accuracy"],
        return_train_score=False
    )

    results.append({
        "feature_set": name,
        "accuracy_mean": round(scores["test_accuracy"].mean(), 4),
        "accuracy_std": round(scores["test_accuracy"].std(), 4),
        "balanced_accuracy_mean": round(
            scores["test_balanced_accuracy"].mean(), 4
        ),
        "balanced_accuracy_std": round(
            scores["test_balanced_accuracy"].std(), 4
        )
    })

result = pd.DataFrame(results)

print("========== BACKGROUND SHORTCUT TEST ==========")
print(result.to_string(index=False))

output = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_background_shortcut_test.csv"
)

result.to_csv(output, index=False)

print("\nSaved:", output)
print("Original images and manifests were not changed.")
