from pathlib import Path
import cv2
import numpy as np
import csv

ROOT = Path(".")
image_path = ROOT / "datasets/external/pillQC/PillQC/pillQC-main/images/chip/chip0032.jpg"

output_dir = ROOT / "reports/dataset_audit/pillqc/fragment_candidate_review"
output_dir.mkdir(parents=True, exist_ok=True)

if not image_path.exists():
    raise FileNotFoundError(f"Original image not found: {image_path}")

# Find the best available existing mask without changing it.
mask_files = [
    p for p in (ROOT / "reports/dataset_audit/pillqc").rglob("*chip0032*")
    if p.is_file()
    and "mask" in str(p).lower()
    and "overlay" not in str(p).lower()
]

mask_files.sort(
    key=lambda p: (
        0 if "centre_seeded" in str(p).lower() else
        1 if "grabcut_masks" in str(p).lower() else 2,
        len(str(p))
    )
)

if not mask_files:
    raise FileNotFoundError(
        "Could not find an existing chip0032 mask. "
        "Please check the grabcut_initialization_comparison folder."
    )

mask_path = mask_files[0]
print("Original image:", image_path)
print("Mask selected:", mask_path)

image = cv2.imread(str(image_path))
mask_image = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)

if image is None or mask_image is None:
    raise RuntimeError("Could not read the original image or selected mask.")

if image.shape[:2] != mask_image.shape[:2]:
    mask_image = cv2.resize(
        mask_image,
        (image.shape[1], image.shape[0]),
        interpolation=cv2.INTER_NEAREST
    )

foreground = (mask_image > 0).astype(np.uint8)
if foreground.sum() == 0:
    raise RuntimeError("The selected mask is empty.")

# Estimate tablet colour from the central, eroded part of the mask.
kernel = np.ones((5, 5), np.uint8)
core = cv2.erode(foreground, kernel, iterations=2)

if core.sum() < 20:
    core = foreground

lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
tablet_pixels = lab[core > 0]
reference_colour = np.median(tablet_pixels, axis=0)

# Look only near the existing tablet boundary.
background = (1 - foreground).astype(np.uint8)
distance = cv2.distanceTransform(background, cv2.DIST_L2, 3)

colour_distance = np.sqrt(
    np.sum((lab - reference_colour) ** 2, axis=2)
)

# Exclude the immediate boundary to avoid simply highlighting mask edges.
nearby = (distance >= 2) & (distance <= 25)

panels = []
rows = []

# Existing mask shown first.
base = image.copy()
base[foreground > 0] = (
    0.55 * base[foreground > 0] +
    0.45 * np.array([0, 0, 255])
).astype(np.uint8)
panels.append(("Existing tablet mask", base))

# Three increasingly permissive colour-similarity tests.
for threshold in (12, 20, 30):
    candidate = nearby & (colour_distance <= threshold)

    # Remove tiny isolated pixel noise, but retain separate fragments.
    candidate_u8 = candidate.astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        candidate_u8, connectivity=8
    )

    cleaned = np.zeros_like(candidate_u8)
    component_count = 0
    candidate_area = 0

    for i in range(1, count):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area >= 3:
            cleaned[labels == i] = 1
            component_count += 1
            candidate_area += area

    overlay = image.copy()
    overlay[foreground > 0] = (
        0.55 * overlay[foreground > 0] +
        0.45 * np.array([0, 0, 255])
    ).astype(np.uint8)

    # Candidate detached fragments are highlighted in yellow.
    overlay[cleaned > 0] = (
        0.25 * overlay[cleaned > 0] +
        0.75 * np.array([0, 255, 255])
    ).astype(np.uint8)

    panels.append((f"Candidate fragments: threshold {threshold}", overlay))

    rows.append({
        "threshold": threshold,
        "candidate_pixels": candidate_area,
        "candidate_components": component_count
    })

# Build a 2x2 comparison sheet.
resized_panels = []
for title, panel in panels:
    panel = cv2.resize(panel, (450, 450))
    cv2.putText(
        panel, title, (8, 24),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55,
        (255, 255, 255), 2, cv2.LINE_AA
    )
    resized_panels.append(panel)

top = np.hstack(resized_panels[:2])
bottom = np.hstack(resized_panels[2:])
sheet = np.vstack([top, bottom])

sheet_path = output_dir / "chip0032_fragment_candidates.jpg"
cv2.imwrite(str(sheet_path), sheet)

csv_path = output_dir / "chip0032_fragment_candidate_counts.csv"
with open(csv_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
    writer.writeheader()
    writer.writerows(rows)

print("\n========== FRAGMENT CANDIDATE AUDIT ==========")
for row in rows:
    print(row)

print("\nContact sheet:", sheet_path)
print("CSV:", csv_path)
print("Original image and existing masks were not changed.")
