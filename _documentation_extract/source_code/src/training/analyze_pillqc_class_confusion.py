import pandas as pd
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

path = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_center_border_analysis.csv"
)

df = pd.read_csv(path)

features = [
    "center_brightness",
    "center_saturation"
]

X = df[features]
y = df["category"]

classes = ["normal", "dirt", "chip"]

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

model = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=2000)
)

predictions = cross_val_predict(
    model,
    X,
    y,
    cv=cv
)

matrix = confusion_matrix(
    y,
    predictions,
    labels=classes
)

print("========== CENTRE-ONLY CONFUSION MATRIX ==========")
print("Rows = actual category")
print("Columns = predicted category")
print(pd.DataFrame(
    matrix,
    index=classes,
    columns=classes
).to_string())

print("\n========== CLASS-WISE PERFORMANCE ==========")
print(classification_report(
    y,
    predictions,
    labels=classes,
    target_names=classes,
    digits=4,
    zero_division=0
))

output = Path(
    "reports/dataset_audit/pillqc/"
    "pillqc_centre_only_confusion_matrix.csv"
)

pd.DataFrame(
    matrix,
    index=classes,
    columns=classes
).to_csv(output)

print("Saved:", output)
print("No images, labels, or manifests changed.")
