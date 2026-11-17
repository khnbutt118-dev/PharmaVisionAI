"""Frame-in / frame-out mask-compliance module (does NOT open the camera).

Usage inside any app that owns the camera:
    mod = MaskComplianceModule("runs/ppe_mask/mask_yolov8n/weights/best.pt")
    annotated, info = mod.process(frame_bgr)
    if info["gate_open"]: ...enable tablet inspection...
"""
from __future__ import annotations
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402
from src.ppe_mask.gate import (Detection, GateConfig, MaskGate, frame_status, WITH_MASK, WITHOUT_MASK,  # noqa: E402
                               INCORRECT, COMPLIANT, NOT_COMPLIANT, UNCERTAIN, NO_FACE)

COLORS = {WITH_MASK: (0, 190, 0), WITHOUT_MASK: (0, 0, 255), INCORRECT: (0, 140, 255)}
LABELS = {WITH_MASK: "MASK OK", WITHOUT_MASK: "NO MASK", INCORRECT: "MASK INCORRECT"}


def banner_for(info: dict):
    status = info["status"]
    if info["gate_open"]:
        return "MASK CHECK PASSED - inspection enabled", (0, 150, 0)
    if status == COMPLIANT:
        return "VERIFYING MASK... hold still", (0, 170, 170)
    if status == NOT_COMPLIANT:
        if info.get("has_incorrect") and not info.get("has_no_mask"):
            return "Please wear your mask correctly before inspection", (0, 100, 230)
        return "NO MASK - please wear your mask before starting inspection", (0, 0, 200)
    if status == UNCERTAIN:
        return "MASK CHECK UNCERTAIN - face the camera, improve lighting", (0, 170, 220)
    return "NO OPERATOR FACE DETECTED - inspection locked", (90, 90, 90)


def draw_overlay(frame, dets, info, show_conf=0.25):
    out = frame
    for d in dets:
        if d.conf < show_conf:
            continue
        x1, y1, x2, y2 = [int(v) for v in d.box]
        color = COLORS.get(d.cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3 if d.cls != WITH_MASK else 2)
        text = f"{LABELS.get(d.cls, d.cls)} {d.conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ty = y1 - 8 if y1 - th - 12 > 0 else y2 + th + 8
        cv2.rectangle(out, (x1, ty - th - 6), (x1 + tw + 8, ty + 4), color, -1)
        cv2.putText(out, text, (x1 + 4, ty - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    msg, col = banner_for(info)
    cv2.rectangle(out, (0, 0), (out.shape[1], 38), col, -1)
    cv2.putText(out, msg, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
    return out


class MaskComplianceModule:
    def __init__(self, weights, conf=0.25, imgsz=640, device="cpu", cfg: GateConfig = GateConfig()):
        from ultralytics import YOLO  # imported lazily
        self.model = YOLO(str(weights))
        self.conf, self.imgsz, self.device = conf, imgsz, device
        self.cfg = cfg
        self.gate = MaskGate(cfg)

    def detect(self, frame):
        res = self.model.predict(frame, conf=self.conf, imgsz=self.imgsz, device=self.device, verbose=False)[0]
        names = res.names
        dets = []
        for b in res.boxes:
            dets.append(Detection(names[int(b.cls[0])], float(b.conf[0]), tuple(float(v) for v in b.xyxy[0].tolist())))
        return dets

    def process(self, frame, draw=True):
        dets = self.detect(frame)
        h, w = frame.shape[:2]
        status = frame_status(dets, self.cfg, frame_area=float(h * w))
        info = self.gate.update(status)
        info["detections"] = dets
        info["has_incorrect"] = any(d.cls == INCORRECT and d.conf >= self.cfg.bad_conf for d in dets)
        info["has_no_mask"] = any(d.cls == WITHOUT_MASK and d.conf >= self.cfg.bad_conf for d in dets)
        out = draw_overlay(frame.copy(), dets, info) if draw else frame
        return out, info
