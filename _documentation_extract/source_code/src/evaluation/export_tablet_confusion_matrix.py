from pathlib import Path
import csv
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    ROOT / "runs" / "tablet_detection"
    / "tablet_yolov8n_pilot" / "weights" / "best.pt"
)

DATA_YAML = (
    ROOT / "datasets" / "tablet_detection"
    / "prepared_stratified" / "data.yaml"
)

model = YOLO(str(MODEL_PATH))

metrics = model.val(
    data=str(DATA_YAML),
    split="test",
    conf=0.25,
    imgsz=640,
    batch=4,
    device="cpu",
    workers=0,
    plots=True,
    project=str(ROOT / "runs" / "tablet_detection"),
    name="tablet_detector_test_matrix",
    exist_ok=True
)

matrix = metrics.confusion_matrix.matrix
class_names = [str(model.names[i]) for i in range(len(model.names))]
labels = class_names + ["background"]

print("\nCONFUSION MATRIX")
print("Rows = actual class | Columns = predicted class")
print("\t" + "\t".join(labels))

for label, row in zip(labels, matrix):
    print(label + "\t" + "\t".join(str(int(value)) for value in row))

output = ROOT / "reports" / "tablet_confusion_matrix.csv"
output.parent.mkdir(parents=True, exist_ok=True)

with open(output, "w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Actual / Predicted"] + labels)

    for label, row in zip(labels, matrix):
        writer.writerow([label] + [int(value) for value in row])

print("\nCSV SAVED:", output)


