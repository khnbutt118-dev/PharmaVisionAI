from pathlib import Path
import random
import shutil

ROOT = Path(__file__).resolve().parents[2]
DATASET = ROOT / "datasets" / "tablet_detection"

SOURCE_IMAGES = DATASET / "images" / "all"
SOURCE_LABELS = DATASET / "labels" / "all"
OUTPUT = DATASET / "prepared_stratified"

SEED = 42
EXTENSIONS = [".jpg", ".jpeg", ".png"]

# Reproducible stratified split.
groups = {
    "normal": list(range(1, 61)) + list(range(81, 102)),
    "chipped": list(range(61, 81)),
}

rng = random.Random(SEED)

split_numbers = {
    "train": [],
    "validation": [],
    "test": [],
}

for condition, numbers in groups.items():
    numbers = numbers.copy()
    rng.shuffle(numbers)

    total = len(numbers)
    train_count = round(total * 0.70)
    validation_count = round(total * 0.15)

    split_numbers["train"].extend(numbers[:train_count])
    split_numbers["validation"].extend(
        numbers[train_count:train_count + validation_count]
    )
    split_numbers["test"].extend(
        numbers[train_count + validation_count:]
    )

# Validate that every source image and annotation exists.
all_numbers = (
    split_numbers["train"]
    + split_numbers["validation"]
    + split_numbers["test"]
)

if len(all_numbers) != 101 or len(set(all_numbers)) != 101:
    raise RuntimeError("Split error: missing or duplicate image numbers.")

for number in all_numbers:
    stem = f"tablet_{number:04d}"

    image_path = next(
        (
            SOURCE_IMAGES / f"{stem}{ext}"
            for ext in EXTENSIONS
            if (SOURCE_IMAGES / f"{stem}{ext}").is_file()
        ),
        None
    )

    label_path = SOURCE_LABELS / f"{stem}.txt"

    if image_path is None:
        raise FileNotFoundError(f"Missing image: {stem}")

    if not label_path.is_file():
        raise FileNotFoundError(f"Missing annotation: {stem}")

# Rebuild only generated split directories.
for split in split_numbers:
    for folder in [
        OUTPUT / "images" / split,
        OUTPUT / "labels" / split,
    ]:
        if folder.exists():
            shutil.rmtree(folder)
        folder.mkdir(parents=True, exist_ok=True)

# Copy matching image-label pairs.
for split, numbers in split_numbers.items():
    for number in numbers:
        stem = f"tablet_{number:04d}"

        image_path = next(
            SOURCE_IMAGES / f"{stem}{ext}"
            for ext in EXTENSIONS
            if (SOURCE_IMAGES / f"{stem}{ext}").is_file()
        )

        label_path = SOURCE_LABELS / f"{stem}.txt"

        shutil.copy2(
            image_path,
            OUTPUT / "images" / split / image_path.name
        )
        shutil.copy2(
            label_path,
            OUTPUT / "labels" / split / label_path.name
        )

# Write YOLO dataset configuration.
yaml_path = OUTPUT / "data.yaml"
yaml_path.write_text(
    f'path: "{OUTPUT.as_posix()}"\n'
    'train: images/train\n'
    'val: images/validation\n'
    'test: images/test\n'
    'names:\n'
    '  0: tablet\n',
    encoding="utf-8"
)

# Print condition distribution for verification.
print("\nPHARMASIGHT AI - STRATIFIED DATASET SPLIT")
print("=" * 55)

for split, numbers in split_numbers.items():
    chipped = sum(61 <= n <= 80 for n in numbers)
    normal = len(numbers) - chipped

    print(
        f"{split:12} | Normal: {normal:2} | "
        f"Chipped: {chipped:2} | Total: {len(numbers):2}"
    )

print("=" * 55)
print("Total images:", len(all_numbers))
print("Random seed:", SEED)
print("Dataset YAML:", yaml_path)
print("Original images and labels remain unchanged.")
print("Dataset preparation completed successfully.")

