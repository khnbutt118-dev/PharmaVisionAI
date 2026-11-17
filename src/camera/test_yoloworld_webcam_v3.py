import cv2
import time
import torch
from ultralytics import YOLOWorld

MODEL_NAME = "yolov8s-worldv2.pt"
CAMERA_INDEX = 0
IMAGE_SIZE = 416
CONFIDENCE = 0.10


def main():
    print("=" * 60)
    print("PHARMASIGHT AI - TABLET DETECTION V3")
    print("=" * 60)

    # Limit CPU thread overhead on this CPU-only computer.
    torch.set_num_threads(4)

    print("\nLoading YOLO-World...")
    model = YOLOWorld(MODEL_NAME)

    # Detect tablets only. Do not ask the detector to diagnose damage.
    model.set_classes(["tablet"])

    print("Detection class: tablet")
    print("Opening webcam...")

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\nCamera ready. Press Q to quit.")

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Unable to read camera frame.")
                break

            start_time = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=0.45,
                device="cpu",
                verbose=False
            )

            result = results[0]

            inference_ms = (
                time.perf_counter() - start_time
            ) * 1000

            display = result.plot()

            if result.boxes is not None and len(result.boxes) > 0:
                best_index = int(result.boxes.conf.argmax().item())
                best_box = result.boxes[best_index]

                confidence = float(best_box.conf.item())

                x1, y1, x2, y2 = map(
                    int, best_box.xyxy[0].tolist()
                )

                label = f"TABLET | Confidence: {confidence:.1%}"

                cv2.putText(
                    display,
                    label,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

            else:
                cv2.putText(
                    display,
                    "Searching for tablet...",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                display,
                f"Inference: {inference_ms:.0f} ms",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 0),
                2
            )

            cv2.imshow(
                "PharmaSight AI - Tablet Detection V3",
                display
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nCamera closed.")


if __name__ == "__main__":
    main()
