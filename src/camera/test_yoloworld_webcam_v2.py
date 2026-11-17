import cv2
import time
from ultralytics import YOLOWorld

MODEL_NAME = "yolov8s-worldv2.pt"
CAMERA_INDEX = 0

# Additional descriptions for damaged and partially broken tablets.
PROMPTS = [
    "tablet",
    "pill",
    "broken tablet",
    "half tablet",
    "chipped pill",
    "tablet fragment",
]

IMAGE_SIZE = 512
CONFIDENCE = 0.08


def main():
    print("=" * 60)
    print("PHARMASIGHT AI - TABLET DETECTION V2")
    print("=" * 60)

    print("\nLoading YOLO-World...")
    model = YOLOWorld(MODEL_NAME)
    model.set_classes(PROMPTS)

    print("Prompts:", PROMPTS)
    print("Opening webcam...")

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nCamera ready. Press Q to quit.")

    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Camera frame unavailable.")
                break

            # Track detections across frames to help maintain object IDs.
            results = model.track(
                source=frame,
                persist=True,
                tracker="bytetrack.yaml",
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=0.45,
                device="cpu",
                verbose=False
            )

            result = results[0]
            display = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                best_index = int(result.boxes.conf.argmax().item())
                best_box = result.boxes[best_index]

                confidence = float(best_box.conf.item())
                class_id = int(best_box.cls.item())
                label = model.names[class_id]

                x1, y1, x2, y2 = map(
                    int, best_box.xyxy[0].tolist()
                )

                print(
                    f"\rDetected: {label:18s} "
                    f"Confidence: {confidence:.1%} "
                    f"Box: ({x1},{y1})-({x2},{y2})",
                    end=""
                )

            else:
                cv2.putText(
                    display,
                    "Searching for tablet...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            current_time = time.time()
            elapsed = current_time - previous_time
            previous_time = current_time

            fps = 1.0 / max(elapsed, 0.0001)

            cv2.putText(
                display,
                f"Processing FPS: {fps:.1f}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "PharmaSight AI - Tablet Detection V2",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\n\nCamera closed.")


if __name__ == "__main__":
    main()
