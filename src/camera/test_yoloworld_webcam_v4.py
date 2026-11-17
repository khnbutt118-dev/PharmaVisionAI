import cv2
import time
import torch
from ultralytics import YOLOWorld

MODEL_NAME = "yolov8s-worldv2.pt"
CAMERA_INDEX = 0
IMAGE_SIZE = 512
CONFIDENCE = 0.08

PROMPTS = [
    "pill",
    "medicine tablet",
    "pharmaceutical tablet",
    "tablet"
]


def main():
    torch.set_num_threads(4)

    print("=" * 60)
    print("PHARMASIGHT AI - TABLET DETECTION V4")
    print("=" * 60)

    print("\nLoading model...")
    model = YOLOWorld(MODEL_NAME)
    model.set_classes(PROMPTS)

    print("Search prompts:", PROMPTS)

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
                break

            start = time.perf_counter()

            results = model.predict(
                source=frame,
                imgsz=IMAGE_SIZE,
                conf=CONFIDENCE,
                iou=0.45,
                device="cpu",
                verbose=False
            )

            inference_ms = (time.perf_counter() - start) * 1000

            display = frame.copy()
            boxes = results[0].boxes
            count = 0

            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(
                        int, box.xyxy[0].tolist()
                    )

                    confidence = float(box.conf.item())

                    # All candidate labels are shown neutrally.
                    cv2.rectangle(
                        display,
                        (x1, y1),
                        (x2, y2),
                        (0, 220, 0),
                        2
                    )

                    cv2.putText(
                        display,
                        f"TABLET {confidence:.0%}",
                        (x1, max(25, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 220, 0),
                        2
                    )

                    count += 1

            if count == 0:
                cv2.putText(
                    display,
                    "No tablet detected",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            cv2.putText(
                display,
                f"Inference: {inference_ms:.0f} ms | Objects: {count}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            cv2.imshow("PharmaSight AI - V4", display)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("\nCamera closed.")


if __name__ == "__main__":
    main()
