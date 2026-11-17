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
    if len(points)<18: return None
    pts=np.asarray(points,dtype=np.int32).reshape(-1,1,2)
    closed=np.vstack([pts,pts[:1]])
    perimeter=cv2.arcLength(closed,True)
    if perimeter<100: return None
    approx=cv2.approxPolyDP(closed,0.035*perimeter,True)
    x,y,w,h=cv2.boundingRect(closed); area=abs(cv2.contourArea(closed))
    circularity=4*math.pi*area/max(1.0,perimeter*perimeter)
    if w<28 or h<28: return None
    if circularity>0.72 and len(approx)>=5: return ("CIRCLE",(x+w//2,y+h//2,max(w,h)//2))
    if len(approx)==3: return ("TRIANGLE",approx.reshape(-1,2))
    if len(approx)==4: return ("RECTANGLE",approx.reshape(-1,2))
    if len(approx)==5: return ("PENTAGON",approx.reshape(-1,2))
    return None

def render_shape(canvas,shape,color,thick):
    if not shape:return
    name,data=shape
    if name=="CIRCLE": cv2.circle(canvas,(data[0],data[1]),data[2],color,thick,cv2.LINE_AA)
    else: cv2.polylines(canvas,[np.asarray(data,dtype=np.int32)],True,color,thick,cv2.LINE_AA)

def neon_line(canvas,p1,p2,color,thick):
    cv2.line(canvas,p1,p2,color,thick+13,cv2.LINE_AA)
    cv2.line(canvas,p1,p2,color,thick+5,cv2.LINE_AA)
    cv2.line(canvas,p1,p2,(245,245,255),max(1,thick//2),cv2.LINE_AA)

def draw_molecule(frame,angle):
    # Stylized illustrative molecule, not a chemically accurate model.
    nodes=np.array([[-105,0,0],[-55,-70,25],[15,-85,-15],[90,-35,30],
                    [85,55,-25],[5,85,20],[-70,65,-20],[0,0,90]],float)
    a=math.radians(angle); rot=np.array([[math.cos(a),0,math.sin(a)],[0,1,0],[-math.sin(a),0,math.cos(a)]])
    pts=[]
    for x,y,z in nodes@rot.T:
        scale=420/(420+z); pts.append((int(W/2+x*scale),int(H/2+20+y*scale)))
    for i,j in [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6),(6,0),(0,7),(2,7),(4,7)]:
        cv2.line(frame,pts[i],pts[j],(255,180,70),3,cv2.LINE_AA)
    for i,p in enumerate(pts): cv2.circle(frame,p,17 if i==7 else 13,(255,90,220) if i==7 else (70,240,255),-1,cv2.LINE_AA)
    cv2.putText(frame,"MOLECULAR INTERACTION | VISUAL DEMO",(W//2-205,150),cv2.FONT_HERSHEY_SIMPLEX,.52,(225,245,255),1,cv2.LINE_AA)

def hud(frame,canvas,gesture,hand,fps):
    out=frame.copy(); mask=np.max(canvas,axis=2)>0
    if np.any(mask):
        glow=cv2.GaussianBlur(canvas,(0,0),9); out=cv2.addWeighted(out,1,glow,.32,0)
        blend=cv2.addWeighted(out,.30,canvas,.70,0); out[mask]=blend[mask]
    layer=out.copy(); cv2.rectangle(layer,(0,0),(W,62),(12,10,28),-1); out=cv2.addWeighted(layer,.86,out,.14,0)
    cv2.putText(out,"PHARMASIGHT / GESTURE INTERACTION STUDIO",(15,24),cv2.FONT_HERSHEY_SIMPLEX,.52,(235,245,255),1,cv2.LINE_AA)
    cv2.putText(out,f"MODE: {APP['mode']}",(15,49),cv2.FONT_HERSHEY_SIMPLEX,.48,(255,210,70),1,cv2.LINE_AA)
    cv2.putText(out,f"{fps:.0f} FPS",(W-90,26),cv2.FONT_HERSHEY_SIMPLEX,.5,(100,255,220),1,cv2.LINE_AA)
    if APP["mode"]=="DRAW":
        for i,(_,color) in enumerate(COLORS):
            cx,cy=W-33,145+i*42; cv2.circle(out,(cx,cy),16,color,-1,cv2.LINE_AA)
            if APP["color"]==i: cv2.circle(out,(cx,cy),21,(255,255,255),2,cv2.LINE_AA)
    panel=out.copy(); cv2.rectangle(panel,(14,72),(W-75,128),(12,12,28),-1); out=cv2.addWeighted(panel,.80,out,.20,0)
    cv2.putText(out,APP["recognition_status"][:105],(24,94),cv2.FONT_HERSHEY_SIMPLEX,.42,(80,235,255),1,cv2.LINE_AA)
    if APP["recognized_text"]: cv2.putText(out,"RESULT: "+APP["recognized_text"][:85],(24,116),cv2.FONT_HERSHEY_SIMPLEX,.42,(245,245,255),1,cv2.LINE_AA)
    cv2.putText(out,f"HAND: {hand} | GESTURE: {gesture}",(15,H-39),cv2.FONT_HERSHEY_SIMPLEX,.5,(80,255,205),1,cv2.LINE_AA)
    cv2.putText(out,"D Draw  M Mouse  P Present  V Molecule  E Neon  H Shape  R OCR  C Clear  Z Undo  Y Redo  S Save  Q Exit",(15,H-14),cv2.FONT_HERSHEY_SIMPLEX,.34,(220,225,245),1,cv2.LINE_AA)
    return out

def main():
    global PYAUTO
    if not MODEL_PATH.exists(): print("Missing hand model:",MODEL_PATH); return
    cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
    if not cam.isOpened(): cam.release(); cam=cv2.VideoCapture(0)
    if not cam.isOpened(): print("Cannot open webcam"); return
    cam.set(cv2.CAP_PROP_FRAME_WIDTH,W); cam.set(cv2.CAP_PROP_FRAME_HEIGHT,H)
    canvas=np.zeros((H,W,3),np.uint8); undo=[]; redo=[]
    prev=None; smooth=None; prev_gesture="MOVE"; hand="No hand"; stroke=[]
    molecule_angle=0.0; prev_mouse_x=None; last_action=0.0; start=time.perf_counter(); count=0
    opts=mp.tasks.vision.HandLandmarkerOptions(base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,num_hands=1,min_hand_detection_confidence=.5,
        min_hand_presence_confidence=.5,min_tracking_confidence=.5)
    window="PharmaSight AI - Gesture Interaction Studio"; cv2.namedWindow(window,cv2.WINDOW_AUTOSIZE)
    def mouse_cb(event,x,y,flags,param):
        if event!=cv2.EVENT_LBUTTONDOWN:return
        if APP["mode"]=="DRAW" and x>=W-65:
            for i in range(8):
                if abs(y-(145+i*42))<=19: APP["color"]=i; return
        if y<=62:
            if 15<=x<=105:APP["action"]="UNDO"
            elif 110<=x<=200:APP["action"]="REDO"
            elif 205<=x<=295:APP["action"]="CLEAR"
            elif 300<=x<=390:APP["action"]="SAVE"
            elif 395<=x<=515:APP["action"]="RECOGNIZE"
    cv2.setMouseCallback(window,mouse_cb)
    print("Gesture Studio: D draw | M mouse | P presentation | V molecule | E effects | H shape snap")
    print("R OCR | C clear | Z undo | Y redo | S save | 1-8 colors | Q exit")
    print("Optional system control: install pyautogui in this environment")
    try:
        with mp.tasks.vision.HandLandmarker.create_from_options(opts) as tracker:
            while True:
                ok,frame=cam.read()
                if not ok: print("Webcam frame failed"); break
                frame=cv2.resize(cv2.flip(frame,1),(W,H)); rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
                ts=int((time.perf_counter()-start)*1000); result=tracker.detect_for_video(image,ts)
                gesture="NO HAND"; cursor=None
                if result.hand_landmarks:
                    lm=result.hand_landmarks[0]; gesture=classify(lm)
                    if result.handedness and result.handedness[0]:
                        raw=result.handedness[0][0].category_name; hand={"Left":"Right","Right":"Left"}.get(raw,raw)
                    rx,ry=int(lm[8].x*W),int(lm[8].y*H)
                    smooth=(rx,ry) if smooth is None else (int(SMOOTHING*rx+(1-SMOOTHING)*smooth[0]),int(SMOOTHING*ry+(1-SMOOTHING)*smooth[1]))
                    cursor=smooth; pts=[(int(p.x*W),int(p.y*H)) for p in lm]
                    for c in mp.tasks.vision.HandLandmarksConnections.HAND_CONNECTIONS:
                        cv2.line(frame,pts[c.start],pts[c.end],(0,220,150),2,cv2.LINE_AA)
                    for p in pts: cv2.circle(frame,p,3,(0,180,255),-1)
                    mode=APP["mode"]
                    if mode=="DRAW":
                        if gesture in ("POINT","PINCH") and prev_gesture not in ("POINT","PINCH"):
                            snapshot(canvas,undo,redo); stroke=[]
                        if gesture=="POINT":
                            color=COLORS[APP["color"]][1]
                            if prev is not None:
                                if APP["effects"]: neon_line(canvas,prev,cursor,color,APP["pen"])
                                else: cv2.line(canvas,prev,cursor,color,APP["pen"],cv2.LINE_AA)
                            stroke.append(cursor); prev=cursor
                        elif gesture=="PINCH":
                            cv2.circle(canvas,cursor,APP["eraser"],(0,0,0),-1,cv2.LINE_AA)
                            cv2.circle(frame,cursor,APP["eraser"],(0,0,255),2,cv2.LINE_AA); prev=cursor
                        else:
                            prev=None
                            if prev_gesture=="POINT" and APP["shape_snap"]:
                                shape=detect_shape(stroke)
                                if shape:
                                    # The original freehand remains; snap adds a clean outline on top.
                                    render_shape(canvas,shape,COLORS[APP["color"]][1],max(2,APP["pen"]))
                                    APP["recognition_status"]="Smart shape: "+shape[0]
                            stroke=[]
                    elif mode=="MOUSE":
                        try:
                            if PYAUTO is None:
                                import pyautogui as pg
                                PYAUTO=pg; pg.FAILSAFE=False
                            sw,sh=PYAUTO.size(); sx=int(np.interp(cursor[0],[0,W],[0,sw-1])); sy=int(np.interp(cursor[1],[0,H],[0,sh-1]))
                            PYAUTO.moveTo(sx,sy,duration=0)
                            if gesture=="PINCH" and time.time()-last_action>.5: PYAUTO.click(); last_action=time.time()
                            if gesture=="TWO": PYAUTO.scroll(1 if lm[12].y<lm[8].y else -1)
                            if gesture!="FIST": PYAUTO.mouseUp()
                            elif prev_gesture!="FIST": PYAUTO.mouseDown()
                        except Exception as e: APP["recognition_status"]="Mouse mode needs pyautogui: "+str(e)[:65]
                    elif mode=="PRESENT":
                        try:
                            if PYAUTO is None:
                                import pyautogui as pg
                                PYAUTO=pg
                            if prev_mouse_x is not None and time.time()-last_action>.8:
                                dx=cursor[0]-prev_mouse_x
                                if dx>45: PYAUTO.press("right"); last_action=time.time()
                                elif dx < -45: PYAUTO.press("left"); last_action=time.time()
                            if gesture=="FIST" and time.time()-last_action>.8: PYAUTO.press("esc"); last_action=time.time()
                        except Exception as e: APP["recognition_status"]="Presentation mode needs pyautogui: "+str(e)[:60]
                    elif mode=="MOLECULE" and gesture=="POINT":
                        molecule_angle=(molecule_angle+(cursor[0]-W/2)*.0008)%360
                    prev_mouse_x=cursor[0]
                    cv2.circle(frame,cursor,9,(255,255,255),2,cv2.LINE_AA); cv2.circle(frame,cursor,3,(70,240,255),-1)
                else:
                    prev=None; smooth=None; prev_mouse_x=None
                    if APP["mode"]=="MOUSE" and PYAUTO is not None:
                        try:PYAUTO.mouseUp()
                        except Exception:pass
                prev_gesture=gesture
                action=APP["action"]; APP["action"]=None
                key=cv2.waitKey(1)&0xFF
                if key==ord('d'):APP["mode"]="DRAW";prev=None
                elif key==ord('m'):APP["mode"]="MOUSE";prev=None
                elif key==ord('p'):APP["mode"]="PRESENT";prev=None
                elif key==ord('v'):APP["mode"]="MOLECULE";prev=None
                elif key==ord('e'):APP["effects"]=not APP["effects"]
                elif key==ord('h'):APP["shape_snap"]=not APP["shape_snap"]
                elif key==ord('c'):action="CLEAR"
                elif key==ord('z'):action="UNDO"
                elif key==ord('y'):action="REDO"
                elif key==ord('s'):action="SAVE"
                elif key==ord('r'):action="RECOGNIZE"
                elif ord('1')<=key<=ord('8'):APP["color"]=key-ord('1')
                if action=="UNDO" and undo:redo.append(canvas.copy());canvas=undo.pop()
                elif action=="REDO" and redo:undo.append(canvas.copy());canvas=redo.pop()
                elif action=="CLEAR":
                    if np.any(canvas):snapshot(canvas,undo,redo);canvas[:]=0
                    prev=None
                elif action=="SAVE":save_canvas(canvas)
                elif action=="RECOGNIZE":request_ocr(canvas)
                count+=1; elapsed=time.perf_counter()-start; fps=count/elapsed if elapsed else 0
                if APP["mode"]=="MOLECULE":draw_molecule(frame,molecule_angle)
                cv2.imshow(window,hud(frame,canvas,gesture,hand,fps))
                if key in (ord('q'),27):break
    finally:
        if PYAUTO is not None:
            try:PYAUTO.mouseUp()
            except Exception:pass
        cam.release();cv2.destroyAllWindows();print("Webcam released. Studio closed.")

if __name__=="__main__":main()
