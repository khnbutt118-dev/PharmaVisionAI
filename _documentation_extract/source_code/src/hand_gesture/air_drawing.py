import time
from pathlib import Path
from datetime import datetime

import cv2
import mediapipe as mp
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "src" / "hand_gesture" / "assets" / "hand_landmarker.task"
SAVE_DIR = ROOT / "reports" / "hand_gesture"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
SMOOTHING = 0.35

def finger_extended(landmarks, tip, pip):
    return landmarks[tip].y < landmarks[pip].y

def classify_gesture(landmarks):
    index_up = finger_extended(landmarks, 8, 6)
    middle_up = finger_extended(landmarks, 12, 10)
    ring_up = finger_extended(landmarks, 16, 14)
    pinky_up = finger_extended(landmarks, 20, 18)

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "DRAW"

    if index_up and middle_up and ring_up and pinky_up:
        return "PAUSE"

    return "MOVE"

def draw_glow(canvas, start, end):
    # Neon-style virtual string
    cv2.line(canvas, start, end, (255, 80, 0), 12, cv2.LINE_AA)
    cv2.line(canvas, start, end, (255, 180, 40), 5, cv2.LINE_AA)
    cv2.line(canvas, start, end, (255, 255, 255), 2, cv2.LINE_AA)

def main():
    if not MODEL_PATH.exists():
        print("ERROR: Hand landmark model not found:", MODEL_PATH)
        return

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

    BaseOptions = mp.tasks.BaseOptions
    RunningMode = mp.tasks.vision.RunningMode
    HandLandmarker = mp.tasks.vision.HandLandmarker

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    canvas = None
    previous_point = None
    smoothed_point = None
    start_time = time.perf_counter()
    frame_count = 0

    print("\nPharmaSight AI - Air Drawing")
    print("--------------------------------")
    print("Point with index finger to draw.")
    print("Open palm to pause.")
    print("C = Clear | S = Save | Q = Exit")

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR: Webcam frame unavailable.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]

                if canvas is None:
                    canvas = np.zeros((height, width, 3), dtype=np.uint8)

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int((time.perf_counter() - start_time) * 1000)
                result = landmarker.detect_for_video(mp_image, timestamp)

                gesture = "NO HAND"
                label = "No hand detected"
                drawing = False

                if result.hand_landmarks:
                    landmarks = result.hand_landmarks[0]
                    gesture = classify_gesture(landmarks)

                    # Correct the displayed handedness for this mirrored
                    # webcam configuration, based on the user's two-hand test.
                    label = "Unknown hand"
                    if result.handedness and result.handedness[0]:
                        raw_label = result.handedness[0][0].category_name
                        label = {
                            "Left": "Right",
                            "Right": "Left"
                        }.get(raw_label, raw_label)

                    # Draw hand skeleton
                    points = [
                        (int(p.x * width), int(p.y * height))
                        for p in landmarks
                    ]

                    connections = (
                        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                    )

                    for connection in connections:
                        cv2.line(
                            frame,
                            points[connection.start],
                            points[connection.end],
                            (0, 255, 100),
                            2,
                            cv2.LINE_AA
                        )

                    for point in points:
                        cv2.circle(frame, point, 4, (0, 180, 255), -1)

                    # Index fingertip is landmark 8
                    raw_x = int(landmarks[8].x * width)
                    raw_y = int(landmarks[8].y * height)

                    if smoothed_point is None:
                        smoothed_point = (raw_x, raw_y)
                    else:
                        sx = int(
                            SMOOTHING * raw_x
                            + (1 - SMOOTHING) * smoothed_point[0]
                        )
                        sy = int(
                            SMOOTHING * raw_y
                            + (1 - SMOOTHING) * smoothed_point[1]
                        )
                        smoothed_point = (sx, sy)

                    cv2.circle(
                        frame, smoothed_point, 10,
                        (255, 255, 255), -1
                    )
                    cv2.circle(
                        frame, smoothed_point, 15,
                        (255, 100, 0), 2
                    )

                    if gesture == "DRAW":
                        drawing = True

                        if previous_point is not None:
                            draw_glow(
                                canvas,
                                previous_point,
                                smoothed_point
                            )

                        previous_point = smoothed_point
                    else:
                        previous_point = None

                    cv2.putText(
                        frame, f"Hand: {label}",
                        (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 255, 0), 2
                    )

                else:
                    previous_point = None
                    smoothed_point = None

                frame_count += 1
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                status_color = (
                    (0, 255, 0) if drawing
                    else (0, 200, 255)
                )

                cv2.putText(
                    frame, f"Gesture: {gesture}",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, status_color, 2
                )

                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (20, 110), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 255), 2
                )

                # Blend virtual canvas over the webcam image
                preview = cv2.addWeighted(
                    frame, 0.75, canvas, 0.85, 0
                )

                # Display camera and drawing canvas side by side
                panel = np.hstack((preview, canvas.copy()))

                panel_height = 600
                panel_width = int(
                    panel.shape[1] * panel_height / panel.shape[0]
                )
                panel = cv2.resize(
                    panel, (panel_width, panel_height)
                )

                cv2.putText(
                    panel, "CAMERA + VIRTUAL STRING",
                    (15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1
                )

                cv2.putText(
                    panel, "AIR DRAWING CANVAS",
                    (width + 15, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1
                )

                cv2.imshow("PharmaSight AI - Air Drawing", panel)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("c"):
                    canvas[:] = 0
                    previous_point = None
                    print("Canvas cleared.")

                elif key == ord("s"):
                    filename = (
                        "air_drawing_"
                        + datetime.now().strftime("%Y%m%d_%H%M%S")
                        + ".png"
                    )
                    save_path = SAVE_DIR / filename
                    cv2.imwrite(str(save_path), canvas)
                    print("Drawing saved:", save_path)

                elif key == ord("q") or key == 27:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released.")

if __name__ == "__main__":
    main()
