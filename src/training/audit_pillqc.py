from pathlib import Path
from collections import Counter
from PIL import Image
import pandas as pd

root = Path("datasets/external/pillQC")

if not root.exists():
    print("ERROR: PillQC folder not found:", root.resolve())
    raise SystemExit(1)

print("\n========== FOLDER STRUCTURE ==========")

for path in sorted(root.rglob("*")):
    if path.is_dir():
        print("[FOLDER]", path.relative_to(root))

print("\n========== IMAGE COUNTS ==========")

for folder in sorted(p for p in root.rglob("*") if p.is_dir()):
    images = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in
        [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"]
    ]

    if images:
        print(folder.relative_to(root), ":", len(images))

print("\n========== IMAGE DIMENSIONS ==========")

extensions = Counter()
dimensions = Counter()

for path in root.rglob("*"):
    if not path.is_file():
        continue

    if path.suffix.lower() in [
        ".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"
    ]:
        extensions[path.suffix.lower()] += 1

        try:
            with Image.open(path) as img:
                dimensions[(img.size, img.mode)] += 1
        except Exception as e:
            print("IMAGE ERROR:", path, e)

print("Extensions:", dict(extensions))
print("\nMost common dimensions and modes:")

for item, count in dimensions.most_common(15):
    print(item, ":", count)

print("\n========== CSV FILES ==========")

for path in root.rglob("*.csv"):
    print("\nCSV:", path.relative_to(root))

    try:
        df = pd.read_csv(path)
        print("Shape:", df.shape)
        print("Columns:", df.columns.tolist())
        print(df.head(8).to_string(index=False))
    except Exception as e:
        print("CSV ERROR:", e)

print("\n========== README FILES ==========")

for path in root.rglob("*"):
    if path.is_file() and path.name.lower().startswith("readme"):
        print("\nREADME:", path.relative_to(root))

        try:
            print(path.read_text(encoding="utf-8", errors="replace")[:10000])
        except Exception as e:
            print("README ERROR:", e)

print("\n========== PILLQC AUDIT COMPLETE ==========")
