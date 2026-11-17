from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGE_DIR = PROJECT_ROOT / "datasets" / "tablet_detection" / "images" / "all"
LABEL_DIR = PROJECT_ROOT / "datasets" / "tablet_detection" / "labels" / "all"

IMAGE_DIR.mkdir(parents=True, exist_ok=True)
LABEL_DIR.mkdir(parents=True, exist_ok=True)

CAMERA_INDEX = 0

boxes = []
drawing = False
start_point = None
current_point = None
frozen_frame = None
saved_count = 0


def next_image_number():
    existing = list(IMAGE_DIR.glob("tablet_*.jpg"))
    numbers = []

    for path in existing:
        try:
            numbers.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    return max(numbers, default=0) + 1


def mouse_callback(event, x, y, flags, param):
    global drawing, start_point, current_point, boxes

    if frozen_frame is None:
        return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        start_point = (x, y)
        current_point = (x, y)

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        current_point = (x, y)

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_point = (x, y)

        x1 = min(start_point[0], end_point[0])
        y1 = min(start_point[1], end_point[1])
        x2 = max(start_point[0], end_point[0])
        y2 = max(start_point[1], end_point[1])

        if x2 - x1 >= 8 and y2 - y1 >= 8:
            boxes.append((x1, y1, x2, y2))

        start_point = None
        current_point = None


def draw_interface(frame):
    display = frame.copy()

    for index, (x1, y1, x2, y2) in enumerate(boxes, start=1):
        cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            display,
            str(index),
            (x1, max(20, y1 - 7)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

    if drawing and start_point and current_point:
        cv2.rectangle(
            display,
            start_point,
            current_point,
            (0, 255, 255),
            2
        )

    instructions = [
        f"TABLETS ANNOTATED: {len(boxes)}",
        "Drag mouse: draw one box per tablet",
        "S: Save    U: Undo    C: Clear",
        "N: Discard frame    Q: Quit"
    ]

    y = 25
    for line in instructions:
        cv2.putText(
            display,
            line,
            (10, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (255, 255, 0),
            2
        )
        y += 27

    return display


def save_annotation(frame):
    global saved_count

    height, width = frame.shape[:2]

    if not boxes:
        print("No boxes drawn. Frame not saved.")
        return

    number = next_image_number()
    filename = f"tablet_{number:04d}"

    image_path = IMAGE_DIR / f"{filename}.jpg"
    label_path = LABEL_DIR / f"{filename}.txt"

    label_lines = []

    for x1, y1, x2, y2 in boxes:
        center_x = ((x1 + x2) / 2) / width
        center_y = ((y1 + y2) / 2) / height
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height

        # YOLO format: class_id, center_x, center_y, width, height
        label_lines.append(
            f"0 {center_x:.6f} {center_y:.6f} "
            f"{box_width:.6f} {box_height:.6f}"
        )

    success = cv2.imwrite(
        str(image_path),
        frame,
        [cv2.IMWRITE_JPEG_QUALITY, 95]
    )

    if not success:
        print("Image could not be saved.")
        return

    label_path.write_text(
        "\n".join(label_lines) + "\n",
        encoding="utf-8"
    )

    saved_count += 1

    print(
        f"SAVED: {filename} | "
        f"Tablets annotated: {len(boxes)}"
    )


def main():
    global boxes, drawing, frozen_frame, saved_count

    print("=" * 60)
    print("PHARMASIGHT AI - WEBCAM TABLET ANNOTATOR")
    print("=" * 60)

    print("\nImages folder:", IMAGE_DIR)
    print("Labels folder:", LABEL_DIR)

    camera = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)

    if not camera.isOpened():
        camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    window = "PharmaSight AI - Draw Tablet Boxes"

    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 1100, 700)
    cv2.setMouseCallback(window, mouse_callback)

    print("\nInstructions:")
    print("SPACE = Freeze frame")
    print("Drag mouse = Draw tablet box")
    print("S = Save annotations")
    print("U = Undo last box")
    print("C = Clear boxes")
    print("N = Discard and continue")
    print("Q = Quit")

    try:
        while True:

            if frozen_frame is None:
                success, frame = camera.read()

                if not success:
                    print("Could not read webcam.")
                    break

                display = frame.copy()

                cv2.putText(
                    display,
                    "SPACE: Freeze frame | Q: Quit",
                    (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

            else:
                display = draw_interface(frozen_frame)

            cv2.imshow(window, display)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            elif key == 32 and frozen_frame is None:
                frozen_frame = frame.copy()
                boxes = []
                print("\nFrame frozen. Draw boxes around tablets.")

            elif frozen_frame is not None:

                if key == ord("s"):
                    save_annotation(frozen_frame)
                    frozen_frame = None
                    boxes = []

                elif key == ord("u"):
                    if boxes:
                        boxes.pop()
                        print("Last box removed.")

                elif key == ord("c"):
                    boxes = []
                    print("All boxes cleared.")

                elif key == ord("n"):
                    frozen_frame = None
                    boxes = []
                    print("Frame discarded.")

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print("\n" + "=" * 60)
    print("ANNOTATION SESSION FINISHED")
    print("Frames saved in this session:", saved_count)
    print("Images folder:", IMAGE_DIR)
    print("Labels folder:", LABEL_DIR)
    print("=" * 60)


if __name__ == "__main__":
    main()
