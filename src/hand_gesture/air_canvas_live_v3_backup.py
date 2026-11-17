"""PharmaSight AI - Gesture Interaction Studio v3.

Features: two-hand tracking, air drawing, neon trails, gesture particles,
shape snapping, virtual mouse, presentation control, interactive molecule demo.
OCR and the color palette have been removed as requested.

Required: mediapipe, opencv-python, numpy
Optional for system mouse/presentation control: pyautogui
"""
import cv2
import mediapipe as mp
import numpy as np
import math
import time
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "src" / "hand_gesture" / "assets" / "hand_landmarker.task"
SAVE_DIR = ROOT / "reports" / "hand_gesture"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
W, H = 960, 540

APP = {
    "mode": "DRAW", "neon": True, "shape_snap": True,
    "status": "Ready | Index finger draws | Pinch erases",
    "last_shape": "", "hand_count": 0,
}
INK = (235, 250, 255)
ACCENT = (80, 245, 210)


def distance(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def classify(lm):
    # Landmark y decreases toward the top of the image.
    fingers = [lm[8].y < lm[6].y, lm[12].y < lm[10].y,
               lm[16].y < lm[14].y, lm[20].y < lm[18].y]
    pinch = distance(lm[4], lm[8]) < 0.047
    if pinch:
        return "PINCH"
    if all(fingers):
        return "OPEN"
    if not any(fingers):
        return "FIST"
    if fingers[0] and fingers[1] and not fingers[2] and not fingers[3]:
        return "TWO"
    if fingers[0] and not any(fingers[1:]):
        return "POINT"
    return "MOVE"


def save_canvas(canvas):
    if not np.any(canvas):
        APP["status"] = "Canvas is empty - draw something first"
        return
    path = SAVE_DIR / f"gesture_art_{datetime.now():%Y%m%d_%H%M%S}.png"
    cv2.imwrite(str(path), canvas)
    APP["status"] = f"Saved artwork: {path.name}"
    print("Saved:", path)


def take_snapshot(canvas, undo, redo):
    if len(undo) >= 30:
        undo.pop(0)
    undo.append(canvas.copy())
    redo.clear()


def neon_line(canvas, p1, p2, color=INK, width=5):
    cv2.line(canvas, p1, p2, color, width + 14, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, color, width + 6, cv2.LINE_AA)
    cv2.line(canvas, p1, p2, (255, 255, 255), max(1, width // 2), cv2.LINE_AA)


def detect_shape(points):
    if len(points) < 16:
        return None
    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    contour = np.vstack([pts, pts[:1]])
    perimeter = cv2.arcLength(contour, True)
    if perimeter < 110:
        return None
    approx = cv2.approxPolyDP(contour, 0.035 * perimeter, True)
    x, y, w, h = cv2.boundingRect(contour)
    area = abs(cv2.contourArea(contour))
    circularity = 4 * math.pi * area / max(1.0, perimeter * perimeter)
    if w < 35 or h < 35 or area < 500:
        return None
    if circularity > 0.72 and len(approx) >= 5:
        return ("CIRCLE", (x + w // 2, y + h // 2, max(w, h) // 2))
    if len(approx) == 3:
        return ("TRIANGLE", approx.reshape(-1, 2))
    if len(approx) == 4:
        return ("RECTANGLE", approx.reshape(-1, 2))
    if len(approx) == 5:
        return ("PENTAGON", approx.reshape(-1, 2))
    return None


def draw_shape(canvas, shape):
    name, data = shape
    color = (80, 255, 220)
    if name == "CIRCLE":
        cv2.circle(canvas, (data[0], data[1]), data[2], color, 4, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [np.asarray(data, np.int32)], True, color, 4, cv2.LINE_AA)
    APP["last_shape"] = name
    APP["status"] = f"Shape recognized: {name}"


class ParticleSystem:
    def __init__(self):
        self.items = []

    def burst(self, x, y, count=34, color=(90, 220, 255)):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(1.4, 5.0)
            self.items.append([float(x), float(y), math.cos(angle) * speed,
                               math.sin(angle) * speed, np.random.randint(18, 38), color])

    def trail(self, x, y, count=3):
        for _ in range(count):
            angle = np.random.uniform(0, 2 * math.pi)
            speed = np.random.uniform(0.3, 1.8)
            self.items.append([float(x), float(y), math.cos(angle) * speed,
                               math.sin(angle) * speed, np.random.randint(10, 20), (255, 180, 90)])

    def render(self, frame):
        alive = []
        overlay = frame.copy()
        for p in self.items:
            x, y, vx, vy, life, color = p
            x += vx; y += vy; vy += 0.035; life -= 1
            if life > 0:
                radius = max(1, min(5, life // 5))
                cv2.circle(overlay, (int(x), int(y)), radius, color, -1, cv2.LINE_AA)
                alive.append([x, y, vx * 0.985, vy * 0.985, life, color])
        self.items = alive[-500:]
        frame[:] = cv2.addWeighted(frame, 0.55, overlay, 0.45, 0)


def draw_hand(frame, lm, label, gesture):
    pts = [(int(p.x * W), int(p.y * H)) for p in lm]
    for connection in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[connection.start], pts[connection.end], (70, 220, 170), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (40, 180, 255), -1, cv2.LINE_AA)
    wrist = pts[0]
    cv2.putText(frame, f"{label}: {gesture}", (max(8, wrist[0] - 45), max(82, wrist[1] - 18)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100, 255, 220), 1, cv2.LINE_AA)
    return pts


def draw_molecule(frame, yaw, pitch, scale):
    # Interactive illustrative particle molecule, not a chemically accurate structure.
    nodes = np.array([[-1.0, 0.0, 0.0], [-.55, -.72, .25], [.15, -.85, -.15],
                      [.9, -.35, .3], [.85, .55, -.25], [.05, .85, .2],
                      [-.7, .65, -.2], [0.0, 0.0, .95]], dtype=float) * 115 * scale
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    rotated = nodes @ (ry @ rx).T
    points = []
    for x, y, z in rotated:
        perspective = 480 / (480 + z)
        points.append((int(W / 2 + x * perspective), int(H / 2 + y * perspective)))
    links = [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,0),(0,7),(2,7),(4,7)]
    for i, j in links:
        cv2.line(frame, points[i], points[j], (220, 130, 70), 3, cv2.LINE_AA)
    for i, point in enumerate(points):
        cv2.circle(frame, point, 17 if i == 7 else 12,
                   (255, 100, 220) if i == 7 else (70, 240, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, point, 20 if i == 7 else 15, (230, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(frame, "INTERACTIVE MOLECULE VISUAL | POINT TO ROTATE | TWO HANDS TO SCALE",
                (W//2 - 300, 92), cv2.FONT_HERSHEY_SIMPLEX, .43, (235, 245, 255), 1, cv2.LINE_AA)


def draw_hud(frame, canvas, fps, gesture_text, particles):
    out = frame.copy()
    mask = np.any(canvas > 0, axis=2)
    if np.any(mask):
        glow = cv2.GaussianBlur(canvas, (0, 0), 9)
        out = cv2.addWeighted(out, 1.0, glow, .30, 0)
        out[mask] = cv2.addWeighted(out, .30, canvas, .70, 0)[mask]
    layer = out.copy()
    cv2.rectangle(layer, (0, 0), (W, 70), (12, 10, 28), -1)
    out = cv2.addWeighted(layer, .86, out, .14, 0)
    cv2.putText(out, "PHARMASIGHT | GESTURE INTERACTION STUDIO", (16, 25),
                cv2.FONT_HERSHEY_SIMPLEX, .55, (235, 245, 255), 1, cv2.LINE_AA)
    cv2.putText(out, f"MODE: {APP['mode']}   NEON: {'ON' if APP['neon'] else 'OFF'}   SHAPES: {'ON' if APP['shape_snap'] else 'OFF'}",
                (16, 52), cv2.FONT_HERSHEY_SIMPLEX, .43, (100, 245, 220), 1, cv2.LINE_AA)
    cv2.putText(out, f"{fps:.0f} FPS | HANDS: {APP['hand_count']}/2", (W - 205, 27),
                cv2.FONT_HERSHEY_SIMPLEX, .45, (100, 255, 220), 1, cv2.LINE_AA)
    panel = out.copy()
    cv2.rectangle(panel, (14, 80), (W - 14, 119), (12, 12, 28), -1)
    out = cv2.addWeighted(panel, .78, out, .22, 0)
    cv2.putText(out, APP['status'][:105], (24, 105), cv2.FONT_HERSHEY_SIMPLEX, .43,
                (80, 235, 255), 1, cv2.LINE_AA)
    cv2.putText(out, gesture_text[:100], (16, H - 40), cv2.FONT_HERSHEY_SIMPLEX, .45,
                (100, 255, 210), 1, cv2.LINE_AA)
    cv2.putText(out, "D Draw | M Mouse | P Present | V Molecule | E Neon | H Shapes | C Clear | Z Undo | Y Redo | S Save | Q Quit",
                (16, H - 13), cv2.FONT_HERSHEY_SIMPLEX, .34, (225, 230, 245), 1, cv2.LINE_AA)
    particles.render(out)
    return out


def main():
    if not MODEL_PATH.exists():
        print("Hand landmark model not found:", MODEL_PATH)
        return
    cam = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cam.isOpened():
        cam.release(); cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Cannot open webcam. Close other apps using the camera and retry.")
        return
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, W)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
    canvas = np.zeros((H, W, 3), np.uint8)
    undo, redo = [], []
    particles = ParticleSystem()
    previous_points = {}
    strokes = {}
    last_gestures = {}
    mouse = None
    last_click = 0.0
    last_slide = 0.0
    swipe_anchor = None
    molecule_yaw, molecule_pitch, molecule_scale = 0.0, 0.0, 1.0
    previous_midpoint = None
    previous_hand_distance = None
    molecule_previous_cursor = None
    start = time.perf_counter(); frame_count = 0
    window = "PharmaSight AI - Gesture Interaction Studio"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, W, H)
    pyautogui = None

    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=.55,
        min_hand_presence_confidence=.55,
        min_tracking_confidence=.55)

    print("Studio started. D=Draw, M=Mouse, P=Presentation, V=Molecule, E=Neon, H=Shapes")
    print("Gestures: index finger draws/moves; pinch erases/clicks; open palm bursts particles.")
    print("Presentation: swipe index finger left/right to change slides; fist sends Escape.")
    print("Molecule: move index finger to rotate; use two hands to scale the model.")

    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as tracker:
            while True:
                ok, frame = cam.read()
                if not ok:
                    APP["status"] = "Webcam frame unavailable"
                    print(APP["status"])
                    break
                frame = cv2.resize(cv2.flip(frame, 1), (W, H))
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
                timestamp = int((time.perf_counter() - start) * 1000)
                result = tracker.detect_for_video(mp_image, timestamp)
                hands = result.hand_landmarks or []
                APP["hand_count"] = len(hands)
                frame_gestures = []
                hand_points = []
                for idx, lm in enumerate(hands):
                    gesture = classify(lm)
                    label = f"HAND {idx + 1}"
                    if result.handedness and idx < len(result.handedness) and result.handedness[idx]:
                        label = result.handedness[idx][0].category_name.upper()
                    pts = draw_hand(frame, lm, label, gesture)
                    hand_points.append((lm, pts, gesture))
                    frame_gestures.append(f"{label}: {gesture}")

                # Gesture particles are independent of the selected interaction mode.
                for idx, (lm, pts, gesture) in enumerate(hand_points):
                    old = last_gestures.get(idx, "MOVE")
                    tip = pts[8]
                    if gesture == "OPEN" and old != "OPEN":
                        particles.burst(tip[0], tip[1], 42)
                        APP["status"] = "Open-palm particle burst"
                    elif gesture == "PINCH":
                        particles.trail(tip[0], tip[1], 5)
                        cv2.circle(frame, tip, 22, (255, 130, 60), 2, cv2.LINE_AA)
                        cv2.circle(frame, tip, 8, (255, 245, 210), -1, cv2.LINE_AA)
                    elif gesture == "POINT":
                        particles.trail(tip[0], tip[1], 2)
                    last_gestures[idx] = gesture

                # Primary-hand interaction; both hands remain tracked and visible.
                if hand_points:
                    lm, pts, gesture = hand_points[0]
                    cursor = pts[8]
                    old = previous_points.get("primary_gesture", "MOVE")
                    if APP["mode"] == "DRAW":
                        if gesture == "POINT":
                            if old != "POINT":
                                take_snapshot(canvas, undo, redo)
                                strokes[0] = []
                            prev = previous_points.get("draw_cursor")
                            if prev is not None:
                                if APP["neon"]: neon_line(canvas, prev, cursor)
                                else: cv2.line(canvas, prev, cursor, INK, 5, cv2.LINE_AA)
                            strokes.setdefault(0, []).append(cursor)
                            previous_points["draw_cursor"] = cursor
                        elif gesture == "PINCH":
                            if old != "PINCH": take_snapshot(canvas, undo, redo)
                            cv2.circle(canvas, cursor, 28, (0, 0, 0), -1, cv2.LINE_AA)
                            previous_points.pop("draw_cursor", None)
                        else:
                            previous_points.pop("draw_cursor", None)
                            if old == "POINT" and APP["shape_snap"]:
                                shape = detect_shape(strokes.get(0, []))
                                if shape: draw_shape(canvas, shape)
                                else: APP["status"] = "No clear shape detected - draw a larger closed shape"
                            strokes[0] = []
                    elif APP["mode"] == "MOUSE":
                        try:
                            if pyautogui is None:
                                import pyautogui as pg
                                pyautogui = pg
                                pyautogui.FAILSAFE = False
                            sw, sh = pyautogui.size()
                            sx = int(np.interp(cursor[0], [0, W], [0, sw - 1]))
                            sy = int(np.interp(cursor[1], [0, H], [0, sh - 1]))
                            pyautogui.moveTo(sx, sy, duration=0)
                            if gesture == "PINCH" and time.time() - last_click > .45:
                                pyautogui.click(); last_click = time.time()
                                APP["status"] = "Virtual mouse: click"
                            if gesture == "TWO": pyautogui.scroll(1 if lm[12].y < lm[8].y else -1)
                            if gesture == "FIST":
                                if old != "FIST": pyautogui.mouseDown()
                            else:
                                pyautogui.mouseUp()
                        except Exception as exc:
                            APP["status"] = "Mouse control error: " + str(exc)[:65]
                    elif APP["mode"] == "PRESENT":
                        try:
                            if pyautogui is None:
                                import pyautogui as pg
                                pyautogui = pg
                                pyautogui.FAILSAFE = False
                            if gesture == "POINT":
                                if swipe_anchor is None: swipe_anchor = cursor[0]
                                delta = cursor[0] - swipe_anchor
                                if abs(delta) > 105 and time.time() - last_slide > 1.0:
                                    pyautogui.press("right" if delta > 0 else "left")
                                    last_slide = time.time(); swipe_anchor = cursor[0]
                                    APP["status"] = "Presentation: next slide" if delta > 0 else "Presentation: previous slide"
                            else:
                                swipe_anchor = cursor[0]
                            if gesture == "FIST" and old != "FIST" and time.time() - last_slide > 1.0:
                                pyautogui.press("esc"); last_slide = time.time()
                                APP["status"] = "Presentation: Escape sent"
                        except Exception as exc:
                            APP["status"] = "Presentation requires pyautogui: " + str(exc)[:55]
                    elif APP["mode"] == "MOLECULE":
                        if gesture == "POINT":
                            if molecule_previous_cursor is not None:
                                molecule_yaw += (cursor[0] - molecule_previous_cursor[0]) * .012
                                molecule_pitch = float(np.clip(molecule_pitch + (cursor[1] - molecule_previous_cursor[1]) * .008, -1.2, 1.2))
                            molecule_previous_cursor = cursor
                        else:
                            molecule_previous_cursor = None
                        if len(hand_points) >= 2:
                            p1 = np.array(hand_points[0][1][8], float)
                            p2 = np.array(hand_points[1][1][8], float)
                            current_distance = np.linalg.norm(p1 - p2)
                            if previous_hand_distance is not None:
                                molecule_scale = float(np.clip(molecule_scale + (current_distance - previous_hand_distance) * .003, .55, 1.8))
                            previous_hand_distance = current_distance
                        else:
                            previous_hand_distance = None
                        APP["status"] = "Move index finger to rotate | Spread/pinch both hands to resize"
                    previous_points["primary_gesture"] = gesture

                # Ensure virtual drag never remains held after hand tracking is lost.
                if not hands:
                    previous_points.pop("draw_cursor", None)
                    previous_points["primary_gesture"] = "MOVE"
                    swipe_anchor = None
                    if pyautogui is not None and APP["mode"] == "MOUSE":
                        try: pyautogui.mouseUp()
                        except Exception: pass

                if APP["mode"] == "MOLECULE":
                    draw_molecule(frame, molecule_yaw, molecule_pitch, molecule_scale)
                frame_count += 1
                elapsed = time.perf_counter() - start
                fps = frame_count / elapsed if elapsed > 0 else 0
                gesture_text = " | ".join(frame_gestures) if frame_gestures else "Show one or two hands to camera"
                display = draw_hud(frame, canvas, fps, gesture_text, particles)
                cv2.imshow(window, display)

                key = cv2.waitKey(1) & 0xFF
                if 65 <= key <= 90:  # Accept uppercase and lowercase shortcuts.
                    key += 32
                if key in (ord('q'), 27): break
                elif key == ord('d'):
                    APP["mode"] = "DRAW"; APP["status"] = "Draw mode | Index finger draws | Pinch erases"
                    previous_points.clear(); strokes.clear()
                elif key == ord('m'):
                    APP["mode"] = "MOUSE"; APP["status"] = "Mouse mode | Index moves | Pinch clicks | Two fingers scroll | Fist drags"
                    previous_points.clear(); molecule_previous_cursor = None
                elif key == ord('p'):
                    APP["mode"] = "PRESENT"; APP["status"] = "Presentation mode | Swipe left/right with index finger | Fist = Escape"
                    swipe_anchor = None; previous_points.clear(); molecule_previous_cursor = None
                elif key == ord('v'):
                    APP["mode"] = "MOLECULE"; APP["status"] = "Molecule mode | Move index to rotate | Two hands resize"
                    previous_points.clear(); molecule_previous_cursor = None
                elif key == ord('e'):
                    APP["neon"] = not APP["neon"]
                    APP["status"] = "Neon brush ON" if APP["neon"] else "Neon brush OFF"
                elif key == ord('h'):
                    APP["shape_snap"] = not APP["shape_snap"]
                    APP["status"] = "Smart shape recognition ON" if APP["shape_snap"] else "Smart shape recognition OFF"
                elif key == ord('c'):
                    if np.any(canvas): take_snapshot(canvas, undo, redo); canvas[:] = 0
                    strokes.clear(); APP["status"] = "Canvas cleared"
                elif key == ord('z'):
                    if undo: redo.append(canvas.copy()); canvas = undo.pop(); APP["status"] = "Undo"
                elif key == ord('y'):
                    if redo: undo.append(canvas.copy()); canvas = redo.pop(); APP["status"] = "Redo"
                elif key == ord('s'): save_canvas(canvas)
    except Exception as exc:
        print("STUDIO ERROR:", type(exc).__name__, str(exc))
        APP["status"] = f"Studio error: {type(exc).__name__}: {exc}"
    finally:
        if pyautogui is not None:
            try: pyautogui.mouseUp()
            except Exception: pass
        cam.release()
        cv2.destroyAllWindows()
        print("Webcam released. Studio closed.")


if __name__ == "__main__":
    main()
