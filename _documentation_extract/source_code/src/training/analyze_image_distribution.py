from pathlib import Path
import pandas as pd

ROOT = Path(".")
OUT = ROOT / "reports" / "dataset_audit"

stats_file = OUT / "image_quality_statistics.csv"
stats = pd.read_csv(stats_file)

stats["image_path"] = stats["image_path"].astype(str).str.strip()

prediction_files = [
    (
        "validation",
        ROOT / "reports" / "training" / "validation_predictions.csv"
    ),
    (
        "test",
        ROOT / "reports" / "evaluation" / "full_test_set_predictions.csv"
    )
]

for split, file in prediction_files:
    print("\n" + "=" * 70)
    print(f"{split.upper()} ERROR ANALYSIS")
    print("=" * 70)

    if not file.exists():
        print("Prediction file not found:", file)
        continue

    pred = pd.read_csv(file)
    pred["image_path"] = pred["image_path"].astype(str).str.strip()
    pred["actual_label"] = pred["actual_label"].str.strip().str.lower()

    if "predicted_label" not in pred.columns:
        pred["predicted_label"] = pred[
            "defective_probability"
        ].apply(
            lambda p: "defective" if p >= 0.25 else "normal"
        )
        print("Using the previously selected validation threshold: 0.25")

    else:
        print("Using recorded test predictions and threshold.")

    pred["predicted_label"] = (
        pred["predicted_label"].str.strip().str.lower()
    )

    merged = pred.merge(
        stats[[
            "image_path",
            "brightness",
            "contrast",
            "saturation",
            "aspect_ratio"
        ]],
        on="image_path",
        how="left"
    )

    errors = merged[
        merged["actual_label"] != merged["predicted_label"]
    ].copy()

    false_negatives = merged[
        (merged["actual_label"] == "defective") &
        (merged["predicted_label"] == "normal")
    ].sort_values("defective_probability")

    false_positives = merged[
        (merged["actual_label"] == "normal") &
        (merged["predicted_label"] == "defective")
    ].sort_values(
        "defective_probability",
        ascending=False
    )

    print("\nTotal images:", len(merged))
    print("Total errors:", len(errors))
    print("False negatives:", len(false_negatives))
    print("False positives:", len(false_positives))

    columns = [
        "image_path",
        "defective_probability",
        "brightness",
        "contrast",
        "saturation"
    ]

    print("\n========== MISSED DEFECTIVE IMAGES ==========")
    if false_negatives.empty:
        print("None")
    else:
        print(false_negatives[columns].head(15).to_string(index=False))

    print("\n========== FALSE ALARMS ON NORMAL IMAGES ==========")
    if false_positives.empty:
        print("None")
    else:
        print(false_positives[columns].head(15).to_string(index=False))

    print("\n========== TEST: CORRECT VS MISSED DEFECTIVE ==========")
    if split == "test":
        actual_defective = merged[
            merged["actual_label"] == "defective"
        ].copy()

        actual_defective["result"] = actual_defective[
            "predicted_label"
        ].map({
            "defective": "correctly_detected",
            "normal": "missed"
        })

        print(
            actual_defective.groupby("result")[
                ["defective_probability", "brightness",
                 "contrast", "saturation"]
            ].mean().round(2).to_string()
        )

    errors.to_csv(
        OUT / f"{split}_prediction_errors_detailed.csv",
        index=False
    )

    false_negatives.to_csv(
        OUT / f"{split}_false_negatives.csv",
        index=False
    )

    false_positives.to_csv(
        OUT / f"{split}_false_positives.csv",
        index=False
    )

print("\n========== ANALYSIS COMPLETE ==========")
print("Detailed reports saved in:", OUT)
