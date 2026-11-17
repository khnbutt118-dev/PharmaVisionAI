import cv2
import mediapipe as mp
import numpy as np
import time
from pathlib import Path
from datetime import datetime
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "src" / "hand_gesture" / "assets" / "hand_landmarker.task"
SAVE_DIR = ROOT / "reports" / "hand_gesture"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0
WIDTH, HEIGHT = 960, 540
TOOLBAR_HEIGHT = 90
SMOOTHING = 0.35
MAX_HISTORY = 20

COLORS = [
    ("WHITE", (255, 255, 255)),
    ("RED", (0, 0, 255)),
    ("GREEN", (0, 255, 0)),
    ("BLUE", (255, 0, 0)),
    ("YELLOW", (0, 255, 255)),
    ("PURPLE", (255, 0, 255)),
    ("CYAN", (255, 255, 0)),
    ("ORANGE", (0, 165, 255)),
]

ui = {
    "action": None,
    "color": 0,
    "pen_size": 7,
    "eraser_size": 35,
    "scale": 0.75,
}

def distance(a, b):
    return np.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)

def classify_gesture(lm):
    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    pinch = distance(lm[4], lm[8]) < 0.045

    if pinch:
        return "ERASE"

    if index_up and not middle_up and not ring_up and not pinky_up:
        return "DRAW"

    if index_up and middle_up and ring_up and pinky_up:
        return "PAUSE"

    return "MOVE"

def draw_line(canvas, p1, p2, color, size):
    cv2.line(canvas, p1, p2, color, size, cv2.LINE_AA)

def save_canvas(canvas):
    mask = np.any(canvas > 0, axis=2)
    if not np.any(mask):
        print("Canvas is empty.")
        return

    white = np.full_like(canvas, 255)
    white[mask] = canvas[mask]

    filename = "air_canvas_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
    path = SAVE_DIR / filename
    cv2.imwrite(str(path), white)
    print("Saved:", path)

def load_recognizer():
    print("\nLoading TrOCR handwritten recognition model...")
    print("The first load requires an internet connection.")

    import torch
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    model_name = "microsoft/trocr-small-handwritten"

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name)
    model.to("cpu")
    model.eval()

    print("TrOCR model loaded successfully.")
    return processor, model

def recognize_canvas(canvas, processor, model):
    gray = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

    points = cv2.findNonZero(mask)

    if points is None:
        return "Canvas is empty."

    x, y, w, h = cv2.boundingRect(points)
    padding = 30

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(canvas.shape[1], x + w + padding)
    y2 = min(canvas.shape[0], y + h + padding)

    crop = mask[y1:y2, x1:x2]
    image = 255 - crop

    # Add a white margin around the writing.
    image = cv2.copyMakeBorder(
        image, 35, 35, 35, 35,
        cv2.BORDER_CONSTANT, value=255
    )

    pil_image = Image.fromarray(image).convert("RGB")

    inputs = processor(
        images=pil_image,
        return_tensors="pt"
    )

    with __import__("torch").no_grad():
        generated = model.generate(
            inputs.pixel_values,
            max_new_tokens=64
        )

    text = processor.batch_decode(
        generated, skip_special_tokens=True
    )[0]

    return text.strip() or "No text recognized."

def mouse_callback(event, x, y, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return

    scale = ui["scale"]
    x = int(x / scale)
    y = int(y / scale)

    # The right half is the canvas panel.
    if x < WIDTH:
        return

    x -= WIDTH

    if y > TOOLBAR_HEIGHT:
        return

    # Color palette.
    if 8 <= y <= 46:
        for i in range(len(COLORS)):
            left = 15 + i * 55
            if left <= x <= left + 38:
                ui["color"] = i
                return

    # Main toolbar buttons.
    if 50 <= y <= 84:
        buttons = [
            ("UNDO", 15, 105),
            ("REDO", 112, 202),
            ("CLEAR", 209, 299),
            ("SAVE", 306, 396),
            ("RECOGNIZE", 403, 520),
        ]

        for action, left, right in buttons:
            if left <= x <= right:
                ui["action"] = action
                return

        # Brush and eraser controls.
        if 530 <= x <= 650:
            ui["action"] = "PEN_MINUS" if x < 590 else "PEN_PLUS"
        elif 655 <= x <= 785:
            ui["action"] = "ERASE_MINUS" if x < 720 else "ERASE_PLUS"

def draw_toolbar():
    bar = np.full(
        (TOOLBAR_HEIGHT, WIDTH, 3),
        (35, 35, 35), dtype=np.uint8
    )

    for i, (name, color) in enumerate(COLORS):
        x = 15 + i * 55
        cv2.rectangle(bar, (x, 10), (x + 38, 42), color, -1)

        thickness = 3 if ui["color"] == i else 1
        cv2.rectangle(
            bar, (x - 2, 8), (x + 40, 44),
            (255, 255, 255), thickness
        )

    buttons = [
        ("UNDO", 15, 105),
        ("REDO", 112, 202),
        ("CLEAR", 209, 299),
        ("SAVE", 306, 396),
        ("RECOGNIZE", 403, 520),
    ]

    for text, left, right in buttons:
        cv2.rectangle(
            bar, (left, 50), (right, 83),
            (65, 65, 65), -1
        )
        cv2.putText(
            bar, text, (left + 7, 72),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48, (255, 255, 255), 1
        )

    cv2.putText(
        bar, f"Pen:{ui['pen_size']}",
        (535, 70), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (255, 255, 255), 1
    )
    cv2.putText(
        bar, f"Eraser:{ui['eraser_size']}",
        (655, 70), cv2.FONT_HERSHEY_SIMPLEX,
        0.48, (255, 255, 255), 1
    )

    cv2.putText(
        bar, "- / +", (535, 84),
        cv2.FONT_HERSHEY_SIMPLEX, 0.35,
        (180, 180, 180), 1
    )
    return bar

def main():
    if not MODEL_PATH.exists():
        print("Hand model missing:", MODEL_PATH)
        return

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera.release()
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print("Cannot open webcam.")
        return

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    canvas = np.zeros((HEIGHT, WIDTH, 3), dtype=np.uint8)
    undo_stack = []
    redo_stack = []

    previous_point = None
    smoothed_point = None
    previous_gesture = "MOVE"
    last_hand_label = "No hand"
    recognized_text = "Write something and press RECOGNIZE."
    processor = None
    recognition_model = None

    start_time = time.perf_counter()
    frame_count = 0
    display_scale = min(1440 / (WIDTH * 2), 1.0)
    ui["scale"] = display_scale

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    window = "PharmaSight AI - Professional Air Canvas"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window, mouse_callback)

    print("\n========================================")
    print("       PHARMASIGHT AI AIR CANVAS")
    print("========================================")
    print("Index finger: Draw")
    print("Thumb + index pinch: Erase")
    print("Open palm: Pause")
    print("Toolbar: Select color and tools")
    print("Keyboard: C Clear | Z Undo | Y Redo")
    print("          S Save | R Recognize | Q Exit")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as landmarker:

            while True:
                success, frame = camera.read()
                if not success:
                    print("Camera frame unavailable.")
                    break

                frame = cv2.flip(frame, 1)
                frame = cv2.resize(frame, (WIDTH, HEIGHT))

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb
                )

                timestamp = int(
                    (time.perf_counter() - start_time) * 1000
                )

                result = landmarker.detect_for_video(
                    mp_image, timestamp
                )

                gesture = "NO HAND"
                drawing = False
                erasing = False

                if result.hand_landmarks:
                    lm = result.hand_landmarks[0]
                    gesture = classify_gesture(lm)

                    if result.handedness and result.handedness[0]:
                        raw = result.handedness[0][0].category_name
                        last_hand_label = {
                            "Left": "Right",
                            "Right": "Left"
                        }.get(raw, raw)

                    points = [
                        (int(p.x * WIDTH), int(p.y * HEIGHT))
                        for p in lm
                    ]

                    for connection in (
                        mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS
                    ):
                        cv2.line(
                            frame,
                            points[connection.start],
                            points[connection.end],
                            (0, 255, 100), 2, cv2.LINE_AA
                        )

                    for point in points:
                        cv2.circle(frame, point, 3, (0, 180, 255), -1)

                    raw_x, raw_y = points[8]

                    if smoothed_point is None:
                        smoothed_point = (raw_x, raw_y)
                    else:
                        smoothed_point = (
                            int(SMOOTHING * raw_x + (1 - SMOOTHING) * smoothed_point[0]),
                            int(SMOOTHING * raw_y + (1 - SMOOTHING) * smoothed_point[1])
                        )

                    cursor = smoothed_point

                    if gesture in ("DRAW", "ERASE") and previous_gesture not in ("DRAW", "ERASE"):
                        if len(undo_stack) >= MAX_HISTORY:
                            undo_stack.pop(0)
                        undo_stack.append(canvas.copy())
                        redo_stack.clear()

                    if gesture == "DRAW":
                        drawing = True

                        if previous_point is not None:
                            draw_line(
                                canvas,
                                previous_point,
                                cursor,
                                COLORS[ui["color"]][1],
                                ui["pen_size"]
                            )

                        previous_point = cursor

                    elif gesture == "ERASE":
                        erasing = True
                        cv2.circle(
                            canvas, cursor,
                            ui["eraser_size"], (0, 0, 0), -1,
                            cv2.LINE_AA
                        )
                        cv2.circle(
                            frame, cursor,
                            ui["eraser_size"], (0, 0, 255), 2
                        )
                        previous_point = cursor

                    else:
                        previous_point = None

                    cv2.circle(
                        frame, cursor, 9,
                        (0, 255, 255), 2
                    )

                else:
                    previous_point = None
                    smoothed_point = None

                previous_gesture = gesture

                # Handle toolbar actions.
                action = ui["action"]
                ui["action"] = None

                if action == "UNDO" or action == "Z":
                    if undo_stack:
                        redo_stack.append(canvas.copy())
                        canvas = undo_stack.pop()

                elif action == "REDO" or action == "Y":
                    if redo_stack:
                        undo_stack.append(canvas.copy())
                        canvas = redo_stack.pop()

                elif action == "CLEAR" or action == "C":
                    if np.any(canvas):
                        undo_stack.append(canvas.copy())
                        canvas[:] = 0
                        redo_stack.clear()
                    previous_point = None

                elif action == "SAVE" or action == "S":
                    save_canvas(canvas)

                elif action == "PEN_MINUS":
                    ui["pen_size"] = max(1, ui["pen_size"] - 1)

                elif action == "PEN_PLUS":
                    ui["pen_size"] = min(30, ui["pen_size"] + 1)

                elif action == "ERASE_MINUS":
                    ui["eraser_size"] = max(8, ui["eraser_size"] - 5)

                elif action == "ERASE_PLUS":
                    ui["eraser_size"] = min(100, ui["eraser_size"] + 5)

                elif action == "RECOGNIZE" or action == "R":
                    if not np.any(canvas):
                        recognized_text = "Canvas is empty."
                    else:
                        try:
                            if processor is None:
                                processor, recognition_model = load_recognizer()

                            recognized_text = recognize_canvas(
                                canvas, processor, recognition_model
                            )
                            print("Recognized text:", recognized_text)

                        except Exception as error:
                            recognized_text = "Recognition error - see terminal."
                            print("Recognition failed:", repr(error))

                frame_count += 1
                elapsed = time.perf_counter() - start_time
                fps = frame_count / elapsed if elapsed > 0 else 0

                status = (
                    "ERASING" if erasing
                    else "DRAWING" if drawing
                    else gesture
                )

                cv2.putText(
                    frame, f"Hand: {last_hand_label}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (255, 255, 0), 2
                )
                cv2.putText(
                    frame, f"Mode: {status}",
                    (20, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 0) if drawing else (0, 150, 255), 2
                )
                cv2.putText(
                    frame, f"FPS: {fps:.1f}",
                    (20, 105), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 255), 1
                )

                # Camera panel header.
                camera_header = np.zeros((TOOLBAR_HEIGHT, WIDTH, 3), dtype=np.uint8)
                camera_header[:] = (25, 25, 25)

                cv2.putText(
                    camera_header, "PHARMASIGHT AI | LIVE HAND TRACKING",
                    (15, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 255, 255), 2
                )

                short_text = recognized_text[:85]
                cv2.putText(
                    camera_header, "Recognized: " + short_text,
                    (15, 65), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (255, 255, 255), 1
                )

                toolbar = draw_toolbar()

                camera_panel = np.vstack((camera_header, frame))
                canvas_panel = np.vstack((toolbar, canvas.copy()))
                combined = np.hstack((camera_panel, canvas_panel))

                display = cv2.resize(
                    combined,
                    None,
                    fx=display_scale,
                    fy=display_scale,
                    interpolation=cv2.INTER_AREA
                )

                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break
                elif key == ord("c"):
                    ui["action"] = "C"
                elif key == ord("z"):
                    ui["action"] = "Z"
                elif key == ord("y"):
                    ui["action"] = "Y"
                elif key == ord("s"):
                    ui["action"] = "S"
                elif key == ord("r"):
                    ui["action"] = "R"
                elif ord("1") <= key <= ord("8"):
                    ui["color"] = key - ord("1")

    finally:
        camera.release()
        cv2.destroyAllWindows()
        print("PharmaSight AI Air Canvas closed. Webcam released.")

if __name__ == "__main__":
    main()
