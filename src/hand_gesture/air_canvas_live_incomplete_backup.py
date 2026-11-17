"""PharmaSight AI - Gesture Interaction Studio.

Preserves the single-webcam air canvas and pinch eraser; adds neon strokes,
smart geometric shape snapping, optional virtual mouse, presentation controls,
and a stylized rotating molecule visualization.

Required: mediapipe, opencv-python, numpy, Pillow, torch, transformers.
Optional system control: pip install pyautogui
"""
import cv2, mediapipe as mp, numpy as np, threading, time, math
from pathlib import Path
from datetime import datetime
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "src" / "hand_gesture" / "assets" / "hand_landmarker.task"
SAVE_DIR = ROOT / "reports" / "hand_gesture"
SAVE_DIR.mkdir(parents=True, exist_ok=True)
W, H = 960, 540
SMOOTHING, MAX_HISTORY = 0.35, 25
COLORS = [("WHITE",(245,245,245)),("RED",(40,55,255)),("GREEN",(65,235,120)),
          ("BLUE",(255,120,55)),("YELLOW",(40,235,255)),("PURPLE",(220,80,255)),
          ("CYAN",(255,225,50)),("ORANGE",(30,150,255))]
APP = {"color":5,"pen":7,"eraser":35,"action":None,"mode":"DRAW","effects":True,
       "shape_snap":True,"recognition_status":"Ready | Draw a word, then press R",
       "recognized_text":"","recognition_busy":False,"lock":threading.Lock()}
PROCESSOR = OCR_MODEL = PYAUTO = None


def dist(a,b): return float(np.hypot(a.x-b.x,a.y-b.y))

def classify(lm):
    pinch = dist(lm[4],lm[8]) < 0.045
    up = [lm[8].y < lm[6].y, lm[12].y < lm[10].y,
          lm[16].y < lm[14].y, lm[20].y < lm[18].y]
    if pinch: return "PINCH"
    if all(up): return "OPEN"
    if not any(up): return "FIST"
    if up[0] and up[1] and not up[2] and not up[3]: return "TWO"
    if up[0] and not any(up[1:]): return "POINT"
    return "MOVE"

def snapshot(canvas, undo, redo):
    if len(undo) >= MAX_HISTORY: undo.pop(0)
    undo.append(canvas.copy()); redo.clear()

def save_canvas(canvas):
    if not np.any(np.max(canvas,axis=2)>0):
        APP["recognition_status"]="Canvas is empty"; return
    path=SAVE_DIR/("gesture_art_"+datetime.now().strftime("%Y%m%d_%H%M%S")+".png")
    out=np.full_like(canvas,255); mask=np.max(canvas,axis=2)>0; out[mask]=canvas[mask]
    cv2.imwrite(str(path),out); APP["recognition_status"]=f"Saved: {path.name}"
    print("Saved:",path)

def recognize_worker(image_snapshot):
    global PROCESSOR, OCR_MODEL
    try:
        with APP["lock"]:
            APP["recognition_status"]="OCR: loading model (first run may take time)..."; APP["recognized_text"]=""
        if PROCESSOR is None or OCR_MODEL is None:
            import torch
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel
            model="microsoft/trocr-small-handwritten"
            PROCESSOR=TrOCRProcessor.from_pretrained(model)
            OCR_MODEL=VisionEncoderDecoderModel.from_pretrained(model).to("cpu").eval()
        with APP["lock"]: APP["recognition_status"]="OCR: preparing handwriting..."
        ink=(np.max(image_snapshot,axis=2)>25).astype(np.uint8)*255
        pts=cv2.findNonZero(ink)
        if pts is None: raise ValueError("No writing found. Draw larger letters first.")
        x,y,w,h=cv2.boundingRect(pts); crop=ink[y:y+h,x:x+w]
        scale=160/max(1,h); crop=cv2.resize(crop,(max(1,int(w*scale)),160),interpolation=cv2.INTER_CUBIC)
        prepared=cv2.copyMakeBorder(255-crop,45,45,60,60,cv2.BORDER_CONSTANT,value=255)
        pil=Image.fromarray(prepared).convert("RGB")
        with APP["lock"]: APP["recognition_status"]="OCR: running inference..."
        import torch
        inputs=PROCESSOR(images=pil,return_tensors="pt")
        with torch.no_grad(): output=OCR_MODEL.generate(inputs.pixel_values,max_new_tokens=64,num_beams=4)
        text=PROCESSOR.batch_decode(output,skip_special_tokens=True)[0].strip()
        if not text: text="No text recognized. Try larger, clearer, connected letters."
        with APP["lock"]: APP["recognized_text"]=text; APP["recognition_status"]="OCR complete"
        print("Recognized:",text)
    except Exception as e:
        msg=f"{type(e).__name__}: {e}"
        with APP["lock"]: APP["recognition_status"]="OCR failed - see terminal"; APP["recognized_text"]=msg[:100]
        print("OCR ERROR:",msg)
    finally:
        with APP["lock"]: APP["recognition_busy"]=False

def request_ocr(canvas):
    with APP["lock"]:
        if APP["recognition_busy"]: APP["recognition_status"]="OCR already running"; return
        if not np.any(np.max(canvas,axis=2)>0): APP["recognition_status"]="Draw something first"; return
        APP["recognition_busy"]=True; APP["recognition_status"]="OCR starting..."; APP["recognized_text"]=""
    threading.Thread(target=recognize_worker,args=(canvas.copy(),),daemon=True).start()

def detect_shape(points):
    if len(points) < 18:
        return None

    pts = np.asarray(points, dtype=np.int32).reshape(-1, 1, 2)
    closed = np.vstack([pts, pts[:1]])

    perimeter = cv2.arcLength(closed, True)

    if perimeter < 100:
        return None

    approx = cv2.approxPolyDP(closed, 0.035 * perimeter, True)

    x, y, w, h = cv2.boundingRect(closed)
    area = abs(cv2.contourArea(closed))

    circularity = 4 * math.pi * area / max(1.0, perimeter * perimeter)

    if w < 28 or h < 28:
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