from pathlib import Path
from datetime import datetime

target = Path("src/camera/pharmasight_live.py")

if not target.is_file():
    raise FileNotFoundError(f"File not found: {target}")

source = target.read_text(encoding="utf-8")

# Exact replacements: each must occur exactly once.
replacements = [
    (
        'TABLET_CLASSIFIER = ROOT / "models/pillqc_multiclass/best_model.pth"',
        'TABLET_CLASSIFIER = ROOT / "models/combined_binary/best_model.pth"'
    ),
    (
        '        from src.modules.tablet_inspection import TabletInspector',
        '        from src.camera import live_tablet_inspection as tablet_model'
    ),
    (
        '        self.tablet_classifier = TabletInspector(TABLET_CLASSIFIER, device=DEVICE)',
        '''        (
            self.tablet_classifier,
            self.tablet_preprocess,
            self.tablet_defective_index
        ) = tablet_model.load_classifier()'''
    ),
    (
        '''                    prediction = self.tablet_classifier.inspect_image(pil)
                    predicted = str(prediction["predicted_class"]).upper()
                    confidence = float(prediction["confidence"])
                    review = bool(prediction.get("review_required", True))''',
        '''                    predicted, p_defective, confidence = (
                        tablet_model.classify_crop(
                            crop,
                            self.tablet_classifier,
                            self.tablet_preprocess,
                            self.tablet_defective_index
                        )
                    )
                    review = (predicted == "REVIEW REQUIRED")'''
    )
]

# Validate all anchors before changing anything.
for old, new in replacements:
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"Patch stopped: expected one occurrence, found {count}.\n"
            f"Missing or changed section:\n{old}"
        )

# Create backup before writing.
stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
backup = target.with_name(
    f"{target.stem}.backup_{stamp}{target.suffix}"
)
backup.write_text(source, encoding="utf-8")

# Apply the four targeted replacements.
updated = source
for old, new in replacements:
    updated = updated.replace(old, new, 1)

target.write_text(updated, encoding="utf-8")

print("INTEGRATION PATCH APPLIED")
print(f"Updated file: {target}")
print(f"Backup file:  {backup}")
print("Changed only the tablet classifier path, import, initialization, and inference.")
