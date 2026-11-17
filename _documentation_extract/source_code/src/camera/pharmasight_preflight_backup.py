from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]

# Ensure the project root is available for package imports.
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

print("\n========== PHARMASIGHT AI PREFLIGHT ==========")
print("Project root:", ROOT)

# Verify all required model files
required_files = {
    "Gesture model": ROOT / "src/hand_gesture/assets/hand_landmarker.task",
    "Tablet YOLO": ROOT / "runs/tablet_detection/tablet_yolov8n_pilot/weights/best.pt",
    "Tablet classifier": ROOT / "models/pillqc_multiclass/best_model.pth",
    "PPE YOLO": ROOT / "runs/ppe_mask/mask_yolov8n_full/weights/best.pt",
}

for name, path in required_files.items():
    print(f"{name}: {'FOUND' if path.is_file() else 'MISSING'}")
    if not path.is_file():
        raise FileNotFoundError(f"{name} missing: {path}")

# Load the main project's existing package
import src

# Resolve the second project's separate src package
ppe_src = ROOT / "PharmaVision-Module2-PPE-Mask" / "src"
if str(ppe_src) not in src.__path__:
    src.__path__.append(str(ppe_src))

print("\n========== IMPORT TESTS ==========")

from src.hand_gesture import air_canvas_live
print("Gesture Studio helpers: IMPORTED")

from src.modules.tablet_inspection import TabletInspector
print("Three-class tablet classifier: IMPORTED")

from src.ppe_mask.mask_module import MaskComplianceModule
print("PPE compliance module: IMPORTED")

from ultralytics import YOLO
print("Ultralytics YOLO: IMPORTED")

print("\n========== MODEL INITIALIZATION ==========")

print("Loading tablet detector...")
tablet_detector = YOLO(str(required_files["Tablet YOLO"]))
print("Tablet detector classes:", tablet_detector.names)

print("Loading PPE detector...")
ppe_detector = YOLO(str(required_files["PPE YOLO"]))
print("PPE detector classes:", ppe_detector.names)

print("\nLoading three-class tablet classifier...")
tablet_classifier = TabletInspector(
    required_files["Tablet classifier"],
    device="cpu"
)
print("Tablet classifier: INITIALIZED")
print("Classifier labels:", ["normal", "contamination", "chipping"])

print("\n========== PREFLIGHT PASSED ==========")
print("All components imported and model checkpoints loaded.")
print("No webcam was opened.")

