import cv2
import time
from ultralytics import YOLOWorld


MODEL_NAME = "yolov8s-worldv2.pt"
CAMERA_INDEX = 0
IMAGE_SIZE = 640
CONFIDENCE = 0.10

def main():
    print("=" * 60)
    print("PHARMASIGHT AI - YOLO-WORLD TABLET DETECTION TEST")
    print("=" * 60)

    print("\nLoading YOLO-World model...")
    print("The first run may download the model weights.")

    model = YOLOWorld(MODEL_NAME)

    # Ask the open-vocabulary detector to look for tablets and pills.
    model.set_classes(["tablet", "pill"])

    print("Model loaded successfully.")
    print("Opening webcam...")

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check that the camera is connected "
            "and not being used by another application."
        )

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("\nWebcam started.")
    print("Place one tablet in front of the camera.")
    print("Press Q to quit.")

    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Could not read webcam frame.")
                break

            results = model.predict(
                source=frame,
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

                cv2.putText(
                    display,
                    f"Detected: {label} ({confidence:.1%})",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    display,
                    "No tablet detected - adjust position or lighting",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
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

            cv2.imshow("PharmaSight AI - Live Tablet Detection", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nWebcam closed.")


if __name__ == "__main__":
    main()
