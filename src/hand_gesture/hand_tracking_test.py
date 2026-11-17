import time
import cv2
import mediapipe as mp

MODEL_PATH = "src/hand_gesture/assets/hand_landmarker.task"
CAMERA_INDEX = 0

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

def draw_hand(frame, landmarks, label):
    height, width = frame.shape[:2]
    points = []

    for landmark in landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    connections = mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS

    for connection in connections:
        start = points[connection.start]
        end = points[connection.end]
        cv2.line(frame, start, end, (0, 255, 0), 2)

    for point in points:
        cv2.circle(frame, point, 4, (0, 180, 255), -1)

    wrist_x, wrist_y = points[0]
    cv2.putText(
        frame, label, (wrist_x, max(25, wrist_y - 15)),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2
    )

def main():
    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("ERROR: Could not open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("Webcam started. Show your hand to the camera.")
    print("Press Q or ESC to exit.")

    start_time = time.perf_counter()
    frame_count = 0

    try:
        with HandLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()

                if not success:
                    print("ERROR: Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame
                )

                timestamp_ms = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image, timestamp_ms
                )

                hand_count = len(result.hand_landmarks)

                for index, landmarks in enumerate(result.hand_landmarks):
                    label = f"Hand {index + 1}"

                    if index < len(result.handedness):
                        categories = result.handedness[index]
                        if categories:
                            label = categories[0].category_name

                    draw_hand(frame, landmarks, label)

                frame_count += 1
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                cv2.putText(
                    frame,
                    f"Hands detected: {hand_count}",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"FPS: {fps:.1f} | Q / ESC: Exit",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2
                )

                cv2.imshow("PharmaSight AI - Hand Tracking Test", frame)

                key = cv2.waitKey(1) & 0xFF

                if key == ord("q") or key == 27:
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("Webcam released.")

if __name__ == "__main__":
    main()
