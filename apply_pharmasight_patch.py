from pathlib import Path
from datetime import datetime
import shutil

p = Path("src/camera/pharmasight_live.py")
if not p.is_file():
    raise FileNotFoundError(f"Could not find {p}. Run this from the project root.")

source = p.read_text(encoding="utf-8")
original = source

def replace_once(old, new, description):
    global source
    count = source.count(old)
    if count != 1:
        raise RuntimeError(
            f"PATCH STOPPED: expected exactly one match for '{description}', "
            f"found {count}. Original file has not been changed."
        )
    source = source.replace(old, new, 1)
    print("Prepared:", description)

# ---------------------------------------------------------
# 1. Imports and preferred binary checkpoint
# ---------------------------------------------------------
replace_once(
    "from PIL import Image\nfrom ultralytics import YOLO",
    "from PIL import Image\n"
    "from ultralytics import YOLO\n"
    "import torch\n"
    "import torch.nn as nn\n"
    "from torchvision import models, transforms",
    "PyTorch classifier imports"
)

replace_once(
    'TABLET_CLASSIFIER = ROOT / "models/pillqc_multiclass/best_model.pth"',
    'TABLET_CLASSIFIER = ROOT / "models/combined_binary/best_model.pth"',
    "preferred binary checkpoint path"
)

# ---------------------------------------------------------
# 2. Add the original binary classifier implementation
# ---------------------------------------------------------
binary_class = '''

class BinaryTabletInspector:
    """Original PharmaSight ResNet18 normal/defective classifier."""

    def __init__(self, model_path, device="cpu"):
        self.model_path = Path(model_path)
        self.device = torch.device(device)

        checkpoint = torch.load(
            self.model_path,
            map_location=self.device,
            weights_only=False
        )

        if not isinstance(checkpoint, dict):
            raise TypeError("Binary checkpoint must be a dictionary.")

        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(state_dict, dict):
            raise KeyError("Checkpoint has no model_state_dict.")

        self.class_names = [
            str(x).strip().lower()
            for x in checkpoint.get(
                "class_names", ["normal", "defective"]
            )
        ]

        if len(self.class_names) != 2 or set(self.class_names) != {
            "normal", "defective"
        }:
            raise ValueError(
                f"Unexpected binary class names: {self.class_names}"
            )

        self.defective_index = self.class_names.index("defective")

        self.model = models.resnet18(weights=None)
        self.model.fc = nn.Linear(self.model.fc.in_features, 2)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

        mean = checkpoint.get(
            "normalization_mean", [0.485, 0.456, 0.406]
        )
        std = checkpoint.get(
            "normalization_std", [0.229, 0.224, 0.225]
        )

        # Match the original live-inspection preprocessing.
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        print("Binary classifier loaded:", self.model_path)
        print("Binary classes:", self.class_names)

    @torch.inference_mode()
    def inspect_image(self, image):
        if not isinstance(image, Image.Image):
            raise TypeError("Expected a PIL image.")

        image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]

        p_defective = float(
            probabilities[self.defective_index].item()
        )
        confidence = max(p_defective, 1.0 - p_defective)

        # Preserve the original 40%-60% manual-review band.
        review_required = 0.40 <= p_defective <= 0.60

        if review_required:
            predicted_class = "REVIEW REQUIRED"
        elif p_defective > 0.60:
            predicted_class = "DEFECTIVE"
        else:
            predicted_class = "NORMAL"

        return {
            "predicted_class": predicted_class,
            "confidence": confidence,
            "p_defective": p_defective,
            "review_required": review_required,
            "result_type": "MODEL_PREDICTION_ONLY"
        }

'''

replace_once(
    "\ndef hud(frame, mode, status):",
    binary_class + "\ndef hud(frame, mode, status):",
    "binary inspector class"
)

# ---------------------------------------------------------
# 3. Use binary inspector, not the separate three-class module
# ---------------------------------------------------------
replace_once(
    "        from src.modules.tablet_inspection import TabletInspector\n",
    "",
    "remove three-class inspector import from unified app"
)

replace_once(
    "        self.g = gesture\n",
    "        self.g = gesture\n"
    "        self.g.CAPSULE_API['name'] = 'Paracetamol (Acetaminophen)'\n"
    "        self.particles = self.g.ParticleSystem()\n",
    "reuse original particle system and API display name"
)

replace_once(
    "        self.tablet_classifier = TabletInspector(TABLET_CLASSIFIER, device=DEVICE)",
    "        self.tablet_classifier = BinaryTabletInspector(\n"
    "            TABLET_CLASSIFIER, device=DEVICE\n"
    "        )",
    "load original binary classifier"
)

# ---------------------------------------------------------
# 4. K mode: reset palm interaction when changing modes
# ---------------------------------------------------------
replace_once(
    '        self.last_capsule_pinch = False\n\n'
    '        labels = {',
    '        self.last_capsule_pinch = False\n'
    '        self.capsule_held = False\n'
    '        self.capsule_hold_offset = (0, 0)\n\n'
    '        labels = {',
    "reset capsule palm state on mode switch"
)

replace_once(
    '"CAPSULE": "Capsule explorer | Move with index, pinch to open",',
    '"CAPSULE": "Paracetamol explorer | Open palm picks up, pinch opens/closes",',
    "K mode instructions"
)

# ---------------------------------------------------------
# 5. D mode: restore burst and particle trails
# ---------------------------------------------------------
replace_once(
    '            if self.mode == "DRAW":\n'
    '                if gesture == "POINT":',
    '            if self.mode == "DRAW":\n'
    '                # Restore the original Gesture Studio particle effects.\n'
    '                if gesture == "OPEN" and old != "OPEN":\n'
    '                    self.particles.burst(cursor[0], cursor[1], 42)\n'
    '                elif gesture == "PINCH":\n'
    '                    self.particles.trail(cursor[0], cursor[1], 5)\n'
    '                elif gesture == "POINT":\n'
    '                    self.particles.trail(cursor[0], cursor[1], 2)\n\n'
    '                if gesture == "POINT":',
    "D mode particle burst and trails"
)

replace_once(
    '            frame[mask] = blended[mask]\n\n'
    '        self.status = self.status + (',
    '            frame[mask] = blended[mask]\n\n'
    '        if self.mode == "DRAW":\n'
    '            self.particles.render(frame)\n\n'
    '        self.status = self.status + (',
    "render D mode particles"
)

# ---------------------------------------------------------
# 6. K mode: open palm picks up; release anchors; pinch toggles
# ---------------------------------------------------------
old_capsule = '''            elif self.mode == "CAPSULE":
                self.capsule_center[0] = int(
                    0.85 * self.capsule_center[0] + 0.15 * cursor[0]
                )
                self.capsule_center[1] = int(
                    0.85 * self.capsule_center[1] + 0.15 * cursor[1]
                )
                pinch = gesture == "PINCH"
                if pinch and not self.last_capsule_pinch:
                    self.capsule_target = 0.0 if self.capsule_target > 0.5 else 1.0
                self.last_capsule_pinch = pinch
                self.capsule_open += (
                    self.capsule_target - self.capsule_open
                ) * 0.15
                self.status = "Capsule | Pinch to open or close"
'''

new_capsule = '''            elif self.mode == "CAPSULE":
                # Palm centre uses the wrist and four palm-base landmarks.
                palm_points = np.asarray(
                    [points[i] for i in (0, 5, 9, 13, 17)],
                    dtype=np.float32
                )
                palm = np.mean(palm_points, axis=0)
                palm_x, palm_y = int(palm[0]), int(palm[1])
                cx, cy = self.capsule_center

                near_capsule = (
                    abs(palm_x - cx) < 220 and
                    abs(palm_y - cy) < 135
                )

                # A flat, open palm can pick up and move the floating capsule.
                if gesture == "OPEN":
                    if not self.capsule_held and near_capsule:
                        self.capsule_held = True
                        self.capsule_hold_offset = (
                            cx - palm_x, cy - palm_y
                        )

                    if self.capsule_held:
                        target_x = int(np.clip(
                            palm_x + self.capsule_hold_offset[0],
                            175, W - 175
                        ))
                        target_y = int(np.clip(
                            palm_y + self.capsule_hold_offset[1],
                            150, H - 125
                        ))
                        self.capsule_center[0] = int(
                            0.78 * cx + 0.22 * target_x
                        )
                        self.capsule_center[1] = int(
                            0.78 * cy + 0.22 * target_y
                        )
                        self.status = (
                            "Paracetamol | Floating above open palm"
                        )
                    else:
                        self.status = (
                            "Open palm | Move near the capsule to pick it up"
                        )

                    self.last_capsule_pinch = False

                elif gesture == "PINCH":
                    # Pinch toggles the shell only when the hand is near it.
                    if (
                        not self.last_capsule_pinch
                        and near_capsule
                    ):
                        self.capsule_target = (
                            0.0 if self.capsule_target > 0.5 else 1.0
                        )
                        self.capsule_held = False
                        self.status = (
                            "Paracetamol | API revealed"
                            if self.capsule_target > 0.5
                            else "Paracetamol | Capsule closed"
                        )

                    self.last_capsule_pinch = True

                else:
                    # Closing the hand releases the capsule at its last position.
                    self.capsule_held = False
                    self.last_capsule_pinch = False
                    self.status = (
                        "Capsule anchored | Open palm near it to pick it up"
                    )

                self.capsule_open += (
                    self.capsule_target - self.capsule_open
                ) * 0.15
'''

replace_once(
    old_capsule,
    new_capsule,
    "replace index-finger capsule control with palm interaction"
)

# Release the object if the hand leaves the camera view.
replace_once(
    '            self.previous_gesture = "MOVE"\n'
    '            self.last_capsule_pinch = False\n\n'
    '        if self.mode == "MOLECULE":',
    '            self.previous_gesture = "MOVE"\n'
    '            self.last_capsule_pinch = False\n'
    '            self.capsule_held = False\n\n'
    '        if self.mode == "MOLECULE":',
    "anchor capsule when hand disappears"
)

# ---------------------------------------------------------
# 7. Add blue animated capsule border
# ---------------------------------------------------------
glow_method = '''
    def draw_capsule_glow(self, frame):
        cx, cy = map(int, self.capsule_center)
        separation = int(np.clip(self.capsule_open, 0.0, 1.0) * 150)
        axes = (142 + separation // 2 + 9, 51)

        pulse = 0.5 + 0.5 * math.sin(time.perf_counter() * 4.0)
        thickness = 7 + int(3 * pulse)

        glow = np.zeros_like(frame)
        blue = (255, 170, 55)  # BGR electric blue
        cv2.ellipse(
            glow, (cx, cy), axes, 0, 0, 360,
            blue, thickness, cv2.LINE_AA
        )
        glow = cv2.GaussianBlur(glow, (0, 0), 10)
        frame[:] = cv2.addWeighted(frame, 1.0, glow, 0.48, 0)

        cv2.ellipse(
            frame, (cx, cy), axes, 0, 0, 360,
            (255, 215, 125), 2, cv2.LINE_AA
        )

'''

replace_once(
    "    def process_tablet(self, frame):",
    glow_method + "    def process_tablet(self, frame):",
    "blue animated capsule glow renderer"
)

replace_once(
    '        elif self.mode == "CAPSULE":\n'
    '            self.g.draw_capsule(frame, self.capsule_center, self.capsule_open)\n',
    '        elif self.mode == "CAPSULE":\n'
    '            self.g.draw_capsule(frame, self.capsule_center, self.capsule_open)\n'
    '            self.draw_capsule_glow(frame)\n'
    '            cv2.putText(\n'
    '                frame, "PARACETAMOL (ACETAMINOPHEN)",\n'
    '                (max(12, self.capsule_center[0] - 175), 82),\n'
    '                cv2.FONT_HERSHEY_SIMPLEX, 0.52,\n'
    '                (255, 220, 135), 2, cv2.LINE_AA\n'
    '            )\n'
    '            cv2.putText(\n'
    '                frame, "OPEN PALM: PICK UP / MOVE   |   PINCH: OPEN / CLOSE",\n'
    '                (max(12, self.capsule_center[0] - 235), H - 48),\n'
    '                cv2.FONT_HERSHEY_SIMPLEX, 0.37,\n'
    '                (255, 220, 135), 1, cv2.LINE_AA\n'
    '            )\n',
    "K mode blue outline and correct medicine label"
)

# ---------------------------------------------------------
# 8. T mode: show binary result and probability of defect
# ---------------------------------------------------------
replace_once(
    '                    prediction = self.tablet_classifier.inspect_image(pil)\n'
    '                    predicted = str(prediction["predicted_class"]).upper()\n'
    '                    confidence = float(prediction["confidence"])\n'
    '                    review = bool(prediction.get("review_required", True))\n'
    '                    color = (0, 165, 255) if review else (70, 220, 120)\n\n'
    '                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)\n'
    '                    label = f"{predicted} {confidence:.0%}"\n'
    '                    if review:\n'
    '                        label += " | REVIEW"\n',
    '                    prediction = self.tablet_classifier.inspect_image(pil)\n'
    '                    predicted = str(prediction["predicted_class"]).upper()\n'
    '                    confidence = float(prediction["confidence"])\n'
    '                    p_defective = float(prediction["p_defective"])\n'
    '                    review = bool(prediction["review_required"])\n'
    '                    if review:\n'
    '                        color = (0, 165, 255)\n'
    '                        display_label = "REVIEW REQUIRED"\n'
    '                    elif predicted == "DEFECTIVE":\n'
    '                        color = (40, 40, 235)\n'
    '                        display_label = "DEFECTIVE"\n'
    '                    else:\n'
    '                        color = (70, 220, 120)\n'
    '                        display_label = "NORMAL"\n\n'
    '                    cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)\n'
    '                    label = (\n'
    '                        f"{display_label} | P(defect) {p_defective:.0%}"\n'
    '                    )\n',
    "binary T mode display and defect probability"
)

# ---------------------------------------------------------
# 9. Update K-mode status shown at startup
# ---------------------------------------------------------
replace_once(
    '"CAPSULE": "Capsule explorer | Move with index, pinch to open",',
    '"CAPSULE": "Paracetamol | Open palm picks up; pinch opens/closes",',
    "K mode status text"
)

# ---------------------------------------------------------
# 10. Only write after all patch anchors were verified
# ---------------------------------------------------------
if source == original:
    raise RuntimeError("No changes were prepared.")

backup = p.with_name(
    f"{p.stem}_backup_before_interaction_patch_"
    f"{datetime.now():%Y%m%d_%H%M%S}{p.suffix}"
)
shutil.copy2(p, backup)
p.write_text(source, encoding="utf-8")

print("\\nBackup created:", backup)
print("Updated file:", p)
print("Patch applied. Running syntax check...")

import py_compile
py_compile.compile(str(p), doraise=True)

print("\\nSUCCESS: syntax check passed.")
print("No model checkpoint or original module was modified.")
