"""PharmaSight AI - Gesture Interaction Studio v4.

- Mirrored-camera handedness labels corrected for this user's webcam setup.
- Two-hand tracking and visible landmarks.
- Air drawing, neon brush, shape snapping, particles.
- Interactive aspirin molecular structure: index finger translates, thumb-index
  opening/closing zooms, hand movement rotates; two hands add scale/twist control.
- OCR and virtual mouse mode removed.

Required: mediapipe, opencv-python, numpy
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
    "status": "Ready | D draw | V molecular workspace",
    "last_shape": "", "hand_count": 0,
}
INK = (235, 250, 255)


def dist2d(a, b):
    return float(np.hypot(a.x - b.x, a.y - b.y))


def classify(lm):
    fingers = [lm[8].y < lm[6].y, lm[12].y < lm[10].y,
               lm[16].y < lm[14].y, lm[20].y < lm[18].y]
    pinch = dist2d(lm[4], lm[8]) < 0.047
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


def draw_shape(canvas, shape, rough_points=None):
    name, data = shape
    # Remove the rough gesture trace before drawing the clean detected outline.
    if rough_points:
        # Repaint only a narrow path around the user's rough trace.
        for a, b in zip(rough_points[:-1], rough_points[1:]):
            cv2.line(canvas, a, b, (0, 0, 0), 19, cv2.LINE_AA)
    color = (80, 255, 220)
    if name == "CIRCLE":
        cv2.circle(canvas, (data[0], data[1]), data[2], color, 4, cv2.LINE_AA)
    else:
        cv2.polylines(canvas, [np.asarray(data, np.int32)], True, color, 4, cv2.LINE_AA)
    APP["last_shape"] = name
    APP["status"] = f"Recognized: {name}"


class ParticleSystem:
    def __init__(self):
        self.items = []

    def burst(self, x, y, count=36, color=(90, 220, 255)):
        for _ in range(count):
            a = np.random.uniform(0, 2 * math.pi)
            s = np.random.uniform(1.4, 5.0)
            self.items.append([float(x), float(y), math.cos(a)*s, math.sin(a)*s,
                               np.random.randint(18, 38), color])

    def trail(self, x, y, count=3):
        for _ in range(count):
            a = np.random.uniform(0, 2 * math.pi)
            s = np.random.uniform(.3, 1.8)
            self.items.append([float(x), float(y), math.cos(a)*s, math.sin(a)*s,
                               np.random.randint(10, 20), (255, 180, 90)])

    def render(self, frame):
        alive, overlay = [], frame.copy()
        for x, y, vx, vy, life, color in self.items:
            x += vx; y += vy; vy += .035; life -= 1
            if life > 0:
                cv2.circle(overlay, (int(x), int(y)), max(1, min(5, life//5)), color, -1, cv2.LINE_AA)
                alive.append([x, y, vx*.985, vy*.985, life, color])
        self.items = alive[-500:]
        frame[:] = cv2.addWeighted(frame, .58, overlay, .42, 0)


def draw_hand(frame, lm, label, gesture):
    pts = [(int(p.x * W), int(p.y * H)) for p in lm]
    for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], (70, 220, 170), 2, cv2.LINE_AA)
    for p in pts:
        cv2.circle(frame, p, 3, (40, 180, 255), -1, cv2.LINE_AA)
    wx, wy = pts[0]
    cv2.putText(frame, f"{label}: {gesture}", (max(8, wx-42), max(140, wy-18)),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (100, 255, 220), 1, cv2.LINE_AA)
    return pts


# Aspirin (acetylsalicylic acid), C9H8O4. Heavy-atom connectivity is correct;
# coordinates are an illustrative 3D layout, not an energy-minimized crystal structure.
ASPIRIN_ATOMS = [
    ("C", -1.00,  0.00,  0.00), ("C", -.50, -.87, .10), ("C", .50, -.87, -.08),
    ("C", 1.00, 0.00, .12), ("C", .50, .87, -.10), ("C", -.50, .87, .08),
    ("C", -2.05, 0.00, .18), ("O", -2.62, .88, .28), ("O", -2.65, -.86, -.05),
    ("O", 1.58, -.98, .22), ("C", 2.55, -.72, .30), ("O", 3.15, .18, .38),
    ("C", 2.95, -1.85, .55),
]
ASPIRIN_BONDS = [
    (0,1,1),(1,2,2),(2,3,1),(3,4,2),(4,5,1),(5,0,2),
    (0,6,1),(6,7,2),(6,8,1),(3,9,1),(9,10,1),(10,11,2),(10,12,1)
]
ATOM_COLORS = {"C": (180, 190, 205), "O": (70, 80, 255)}


def draw_aspirin(frame, yaw, pitch, scale, center):
    cx, cy = int(center[0]), int(center[1])
    coords = np.array([a[1:] for a in ASPIRIN_ATOMS], dtype=float) * 76.0 * scale
    cyaw, syaw = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    ry = np.array([[cyaw,0,syaw],[0,1,0],[-syaw,0,cyaw]])
    rx = np.array([[1,0,0],[0,cp,-sp],[0,sp,cp]])
    rot = coords @ (ry @ rx).T
    projected = []
    for x,y,z in rot:
        persp = 620.0 / (620.0 + z)
        projected.append((int(cx+x*persp), int(cy+y*persp)))
    # Bonds first, with parallel strokes for double bonds.
    for i,j,order in ASPIRIN_BONDS:
        p1, p2 = projected[i], projected[j]
        cv2.line(frame, p1, p2, (195, 205, 220), 4, cv2.LINE_AA)
        if order == 2:
            dx,dy = p2[0]-p1[0], p2[1]-p1[1]
            length = max(1.0, math.hypot(dx,dy))
            ox,oy = int(-dy/length*4), int(dx/length*4)
            cv2.line(frame, (p1[0]+ox,p1[1]+oy), (p2[0]+ox,p2[1]+oy),
                     (125, 205, 245), 2, cv2.LINE_AA)
    for idx, atom in enumerate(ASPIRIN_ATOMS):
        element = atom[0]; p = projected[idx]
        radius = 13 if element == "O" else 11
        cv2.circle(frame, p, radius+3, (235,245,255), 1, cv2.LINE_AA)
        cv2.circle(frame, p, radius, ATOM_COLORS[element], -1, cv2.LINE_AA)
        cv2.putText(frame, element, (p[0]-5,p[1]+5), cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(frame, "ASPIRIN | C9H8O4 | 3D STRUCTURE VIEW", (cx-185, max(145,cy-205)),
                cv2.FONT_HERSHEY_SIMPLEX, .48, (120,255,225), 1, cv2.LINE_AA)
    cv2.putText(frame, "INDEX: MOVE + ROTATE   THUMB/INDEX: PINCH TO ZOOM   TWO HANDS: TWIST/SCALE",
                (max(12,cx-310), min(H-52,cy+205)), cv2.FONT_HERSHEY_SIMPLEX, .35,
                (220,230,245), 1, cv2.LINE_AA)


def draw_hud(frame, canvas, fps, gesture_text, particles):
    out = frame.copy()
    mask = np.any(canvas > 0, axis=2)
    if np.any(mask):
        glow = cv2.GaussianBlur(canvas, (0,0), 9)
        out = cv2.addWeighted(out, 1.0, glow, .30, 0)
        blended = cv2.addWeighted(out, .30, canvas, .70, 0)
        out[mask] = blended[mask]
    layer = out.copy()
    cv2.rectangle(layer, (0,0), (W,70), (12,10,28), -1)
    out = cv2.addWeighted(layer,.86,out,.14,0)
    cv2.putText(out,"PHARMASIGHT | JARVIS GESTURE STUDIO",(16,25),cv2.FONT_HERSHEY_SIMPLEX,.55,(235,245,255),1,cv2.LINE_AA)
    cv2.putText(out,f"MODE: {APP['mode']}   NEON: {'ON' if APP['neon'] else 'OFF'}   SHAPES: {'ON' if APP['shape_snap'] else 'OFF'}",
                (16,52),cv2.FONT_HERSHEY_SIMPLEX,.43,(100,245,220),1,cv2.LINE_AA)
    cv2.putText(out,f"{fps:.0f} FPS | HANDS: {APP['hand_count']}/2",(W-205,27),cv2.FONT_HERSHEY_SIMPLEX,.45,(100,255,220),1,cv2.LINE_AA)
    panel=out.copy(); cv2.rectangle(panel,(14,80),(W-14,119),(12,12,28),-1)
    out=cv2.addWeighted(panel,.78,out,.22,0)
    cv2.putText(out,APP['status'][:105],(24,105),cv2.FONT_HERSHEY_SIMPLEX,.43,(80,235,255),1,cv2.LINE_AA)
    cv2.putText(out,gesture_text[:100],(16,H-40),cv2.FONT_HERSHEY_SIMPLEX,.45,(100,255,210),1,cv2.LINE_AA)
    cv2.putText(out,"D Draw | V Aspirin 3D | E Neon | H Shapes | C Clear | Z Undo | Y Redo | S Save | Q Quit",
                (16,H-13),cv2.FONT_HERSHEY_SIMPLEX,.34,(225,230,245),1,cv2.LINE_AA)
    particles.render(out)
    return out


def main():
    if not MODEL_PATH.exists():
        print("Hand landmark model not found:", MODEL_PATH); return
    cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cam.isOpened(): cam.release(); cam=cv2.VideoCapture(0)
    if not cam.isOpened(): print("Cannot open webcam."); return
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,W); cam.set(cv2.CAP_PROP_FRAME_HEIGHT,H)
    canvas=np.zeros((H,W,3),np.uint8); undo=[]; redo=[]; particles=ParticleSystem()
    previous={}; strokes={}; last_gestures={}
    yaw,pitch,mol_scale=0.0,0.0,1.0; mol_center=[W//2,H//2+8]
    prev_index=None; prev_gap=None; prev_two_angle=None; prev_two_span=None
    start=time.perf_counter(); frames=0
    window="PharmaSight AI - JARVIS Gesture Studio"
    cv2.namedWindow(window,cv2.WINDOW_NORMAL); cv2.resizeWindow(window,W,H)
    options=mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO, num_hands=2,
        min_hand_detection_confidence=.55,min_hand_presence_confidence=.55,
        min_tracking_confidence=.55)
    print("Studio running. D=Draw, V=Aspirin 3D, E=Neon, H=Shapes, Q=Quit")
    print("Molecule: index fingertip moves and rotates; thumb-index gap zooms; two hands twist and scale.")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(options) as tracker:
            while True:
                ok,frame=cam.read()
                if not ok:
                    APP['status']="Webcam frame unavailable"; print(APP['status']); break
                frame=cv2.resize(cv2.flip(frame,1),(W,H))
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                mp_image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                timestamp=int((time.perf_counter()-start)*1000)
                result=tracker.detect_for_video(mp_image,timestamp)
                hands=result.hand_landmarks or []; APP['hand_count']=len(hands)
                hand_points=[]; gesture_labels=[]
                for idx,lm in enumerate(hands):
                    gesture=classify(lm)
                    # The user's mirrored webcam feed reports handedness reversed; swap it for display.
                    raw="UNKNOWN"
                    if result.handedness and idx<len(result.handedness) and result.handedness[idx]:
                        raw=result.handedness[idx][0].category_name.upper()
                    label={"LEFT":"RIGHT","RIGHT":"LEFT"}.get(raw,raw)
                    pts=draw_hand(frame,lm,label,gesture)
                    hand_points.append((lm,pts,gesture,label))
                    gesture_labels.append(f"{label}: {gesture}")
                # Gesture-driven visual effects.
                for idx,(lm,pts,gesture,label) in enumerate(hand_points):
                    old=last_gestures.get(idx,"MOVE"); tip=pts[8]
                    if gesture=="OPEN" and old!="OPEN":
                        particles.burst(tip[0],tip[1],42); APP['status']=f"{label} hand: particle burst"
                    elif gesture=="PINCH":
                        particles.trail(tip[0],tip[1],5)
                        cv2.circle(frame,tip,21,(255,130,60),2,cv2.LINE_AA)
                    elif gesture=="POINT": particles.trail(tip[0],tip[1],2)
                    last_gestures[idx]=gesture

                if hand_points:
                    lm,pts,gesture,label=hand_points[0]; cursor=pts[8]
                    old=previous.get('primary_gesture','MOVE')
                    if APP['mode']=="DRAW":
                        if gesture=="POINT":
                            if old!="POINT": take_snapshot(canvas,undo,redo); strokes[0]=[]
                            prev=previous.get('draw_cursor')
                            if prev is not None:
                                if APP['neon']: neon_line(canvas,prev,cursor)
                                else: cv2.line(canvas,prev,cursor,INK,5,cv2.LINE_AA)
                            strokes.setdefault(0,[]).append(cursor); previous['draw_cursor']=cursor
                        elif gesture=="PINCH":
                            if old!="PINCH": take_snapshot(canvas,undo,redo)
                            cv2.circle(canvas,cursor,28,(0,0,0),-1,cv2.LINE_AA)
                        else:
                            previous.pop('draw_cursor',None)
                            if old=="POINT" and APP['shape_snap']:
                                shape=detect_shape(strokes.get(0,[]))
                                if shape: draw_shape(canvas,shape,strokes.get(0,[]))
                                else: APP['status']="No clear shape detected - draw a larger closed shape"
                            strokes[0]=[]
                    elif APP['mode']=="MOLECULE":
                        # Absolute index position moves the model around the full camera workspace.
                        target=(int(np.clip(cursor[0],90,W-90)),int(np.clip(cursor[1],165,H-95)))
                        mol_center[0]=int(.72*mol_center[0]+.28*target[0])
                        mol_center[1]=int(.72*mol_center[1]+.28*target[1])
                        # Thumb-index separation is a continuous pinch-to-zoom control.
                        gap=dist2d(lm[4],lm[8])*W
                        if prev_gap is not None:
                            mol_scale=float(np.clip(mol_scale+(gap-prev_gap)*.009,.42,2.7))
                        prev_gap=gap
                        # Index movement also rotates the 3D structure.
                        if prev_index is not None:
                            dx=cursor[0]-prev_index[0]; dy=cursor[1]-prev_index[1]
                            yaw+=dx*.0045; pitch=float(np.clip(pitch+dy*.0035,-1.35,1.35))
                        prev_index=cursor
                        # Two-hand span scales; the angle between hands twists the model.
                        if len(hand_points)>=2:
                            p1=np.array(hand_points[0][1][8],float); p2=np.array(hand_points[1][1][8],float)
                            vec=p2-p1; span=float(np.linalg.norm(vec)); angle=math.atan2(vec[1],vec[0])
                            if prev_two_span is not None:
                                mol_scale=float(np.clip(mol_scale+(span-prev_two_span)*.006,.42,2.7))
                            if prev_two_angle is not None:
                                da=(angle-prev_two_angle+math.pi)%(2*math.pi)-math.pi
                                yaw+=da
                            prev_two_span=span; prev_two_angle=angle
                        else:
                            prev_two_span=None; prev_two_angle=None
                        APP['status']=f"ASPIRIN 3D | Scale {mol_scale:.2f}x | Move index to position/rotate; open/close pinch to zoom"
                    previous['primary_gesture']=gesture
                else:
                    previous.pop('draw_cursor',None); previous['primary_gesture']='MOVE'
                    prev_index=None; prev_gap=None
                    prev_two_span=None; prev_two_angle=None

                if APP['mode']=="MOLECULE":
                    draw_aspirin(frame,yaw,pitch,mol_scale,mol_center)
                frames+=1; elapsed=time.perf_counter()-start; fps=frames/elapsed if elapsed>0 else 0
                display=draw_hud(frame,canvas,fps," | ".join(gesture_labels) if gesture_labels else "Show one or two hands",particles)
                cv2.imshow(window,display)
                key=cv2.waitKey(1)&0xFF
                if 65<=key<=90: key+=32
                # Q is the only keyboard shortcut that exits the studio.
                if key==ord('q'): break
                elif key==ord('d'):
                    APP['mode']='DRAW'; APP['status']='Draw | Index finger draws | Pinch erases'
                    previous.clear(); strokes.clear(); prev_index=None; prev_gap=None
                elif key==ord('v'):
                    APP['mode']='MOLECULE'; APP['status']='Aspirin 3D | Index moves/rotates; pinch zooms; two hands twist/scale'
                    previous.clear(); prev_index=None; prev_gap=None; prev_two_span=None; prev_two_angle=None
                elif key==ord('e'):
                    APP['neon']=not APP['neon']; APP['status']='Neon brush '+('ON' if APP['neon'] else 'OFF')
                elif key==ord('h'):
                    APP['shape_snap']=not APP['shape_snap']; APP['status']='Smart shapes '+('ON' if APP['shape_snap'] else 'OFF')
                elif key==ord('c'):
                    if np.any(canvas): take_snapshot(canvas,undo,redo); canvas[:]=0
                    strokes.clear(); APP['status']='Canvas cleared'
                elif key==ord('z'):
                    if undo: redo.append(canvas.copy()); canvas=undo.pop(); APP['status']='Undo'
                elif key==ord('y'):
                    if redo: undo.append(canvas.copy()); canvas=redo.pop(); APP['status']='Redo'
                elif key==ord('s'): save_canvas(canvas)
    except Exception as exc:
        print('STUDIO ERROR:',type(exc).__name__,str(exc)); APP['status']=f"Studio error: {type(exc).__name__}: {exc}"
    finally:
        cam.release(); cv2.destroyAllWindows(); print('Webcam released. Studio closed.')


if __name__=="__main__":
    main()
