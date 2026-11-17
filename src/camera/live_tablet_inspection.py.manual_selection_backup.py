from pathlib import Path

import cv2
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ------------------------------------------------------------
# PATHS AND CONFIGURATION
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT / "models" / "combined_binary" / "best_model.pth"
)

CLASS_NAMES = ["normal", "defective"]
IMAGE_SIZE = 224
REVIEW_THRESHOLD = 0.70
INFERENCE_EVERY_N_FRAMES = 5


# ------------------------------------------------------------
# LOAD ORIGINAL COMBINED BINARY MODEL
# ------------------------------------------------------------

def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Original combined binary model not found:\n{MODEL_PATH}"
        )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print(f"Loading model: {MODEL_PATH}")
    print(f"Device: {device}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
        weights_only=True,
    )

    class_names = checkpoint.get("class_names", CLASS_NAMES)

    if class_names != CLASS_NAMES:
        raise ValueError(
            f"Unexpected model classes: {class_names}"
        )

    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, len(CLASS_NAMES))

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    image_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    print("Model loaded successfully.")

    return model, device, image_transform


# ------------------------------------------------------------
# CLASSIFY SELECTED TABLET REGION
# ------------------------------------------------------------

@torch.inference_mode()
def classify_tablet(frame, roi, model, device, image_transform):
    x, y, width, height = roi

    crop = frame[y:y + height, x:x + width]

    if crop.size == 0:
        return None

    rgb_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_crop)

    tensor = image_transform(pil_image).unsqueeze(0).to(device)

    logits = model(tensor)
    probabilities = torch.softmax(logits, dim=1)[0]

    predicted_index = int(torch.argmax(probabilities).item())
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(probabilities[predicted_index].item())

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "normal_probability": float(probabilities[0].item()),
        "defective_probability": float(probabilities[1].item()),
        "review_required": confidence < REVIEW_THRESHOLD,
    }


# ------------------------------------------------------------
# LIVE CAMERA
# ------------------------------------------------------------

def main():
    model, device, image_transform = load_model()

    camera = cv2.VideoCapture(0)

    if not camera.isOpened():
        camera.release()
        raise SystemExit(
            "Could not access webcam. Check Windows camera permissions, "
            "close other camera apps, or try camera index 1."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window_name = "PharmaSight AI - Live Tablet Inspection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    roi = None
    result = None
    frame_count = 0

    print("\nPharmaSight live inspection started.")
    print("Select the tablet with your mouse when prompted.")
    print("R = select another tablet")
    print("Q = quit\n")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read a webcam frame.")
                break

            frame_count += 1
            display = frame.copy()

            # Ask the user to select the tablet region.
            if roi is None:
                cv2.putText(
                    display,
                    "Select a tablet region to begin",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                cv2.imshow(window_name, display)
                cv2.waitKey(1)

                selected = cv2.selectROI(
                    window_name,
                    frame,
                    showCrosshair=True,
                    fromCenter=False,
                )

                x, y, width, height = map(int, selected)

                if width > 0 and height > 0:
                    roi = (x, y, width, height)
                    result = None
                    frame_count = 0
                else:
                    print("No region selected. Press Q to quit or try again.")
                    roi = None

                continue

            # Run inference periodically to reduce CPU load.
            if frame_count % INFERENCE_EVERY_N_FRAMES == 0:
                result = classify_tablet(
                    frame,
                    roi,
                    model,
                    device,
                    image_transform,
                )

            x, y, width, height = roi

            if result is not None:
                if result["review_required"]:
                    box_color = (0, 165, 255)
                    status = "MANUAL REVIEW REQUIRED"
                elif result["predicted_class"] == "defective":
                    box_color = (0, 0, 255)
                    status = "PREDICTED DEFECTIVE"
                else:
                    box_color = (0, 200, 0)
                    status = "PREDICTED NORMAL"

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + width, y + height),
                    box_color,
                    3,
                )

                confidence_text = (
                    f"Confidence: {result['confidence'] * 100:.1f}%"
                )

                cv2.putText(
                    display,
                    status,
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    box_color,
                    2,
                )

                cv2.putText(
                    display,
                    confidence_text,
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                cv2.putText(
                    display,
                    (
                        f"Normal: {result['normal_probability'] * 100:.1f}%"
                        f"   Defective: "
                        f"{result['defective_probability'] * 100:.1f}%"
                    ),
                    (20, 105),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2,
                )

            cv2.rectangle(
                display,
                (x, y),
                (x + width, y + height),
                (255, 255, 0),
                1,
            )

            cv2.putText(
                display,
                "R: select another tablet    Q: quit",
                (20, display.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_name, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("r"):
                roi = None
                result = None

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("PharmaSight live inspection stopped.")


if __name__ == "__main__":
    main()
