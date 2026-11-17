"""Conservative mask-compliance gate logic (pure Python, no model needed).

A detector produces face boxes labelled with_mask / without_mask / mask_weared_incorrect.
This module turns those detections into ONE workflow decision per frame and a stable
open/closed gate over time. Design rules:
  * a low-confidence 'no mask' or 'incorrect' detection is enough to block (bad_conf is low);
  * 'compliant' needs every relevant face to be 'with_mask' above a HIGHER confidence (ok_conf);
  * anything else is UNCERTAIN, never a silent pass;
  * the gate opens only after several consecutive compliant frames and closes quickly.
GMP-inspired awareness aid only. Not a validated personnel-control system.
"""
from __future__ import annotations
from dataclasses import dataclass

WITH_MASK = "with_mask"
WITHOUT_MASK = "without_mask"
INCORRECT = "mask_weared_incorrect"
CLASS_NAMES = [WITH_MASK, WITHOUT_MASK, INCORRECT]

COMPLIANT = "COMPLIANT"
NOT_COMPLIANT = "NOT_COMPLIANT"
UNCERTAIN = "UNCERTAIN"
NO_FACE = "NO_FACE"


@dataclass
class Detection:
    cls: str
    conf: float
    box: tuple  # x1, y1, x2, y2 in pixels


@dataclass
class GateConfig:
    ok_conf: float = 0.60              # min confidence to trust a 'with_mask' face
    bad_conf: float = 0.35             # min confidence to flag 'no mask' / 'incorrect' (deliberately lower)
    min_face_area_frac: float = 0.002  # ignore faces smaller than this fraction of the frame (background people)
    open_after_frames: int = 8         # consecutive COMPLIANT frames before the gate opens
    close_after_frames: int = 3        # consecutive non-compliant frames before the gate closes


def _area(box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def frame_status(dets, cfg: GateConfig = GateConfig(), frame_area: float | None = None) -> str:
    faces = list(dets)
    if frame_area:
        faces = [d for d in faces if _area(d.box) / frame_area >= cfg.min_face_area_frac]
    if not faces:
        return NO_FACE
    if any(d.cls in (WITHOUT_MASK, INCORRECT) and d.conf >= cfg.bad_conf for d in faces):
        return NOT_COMPLIANT
    if all(d.cls == WITH_MASK and d.conf >= cfg.ok_conf for d in faces):
        return COMPLIANT
    return UNCERTAIN


class MaskGate:
    """Temporal gate: opens after N compliant frames, closes after M non-compliant frames."""

    def __init__(self, cfg: GateConfig = GateConfig()):
        self.cfg = cfg
        self.open = False
        self.ok_streak = 0
        self.bad_streak = 0

    def update(self, status: str) -> dict:
        if status == COMPLIANT:
            self.ok_streak += 1
            self.bad_streak = 0
            if self.ok_streak >= self.cfg.open_after_frames:
                self.open = True
        else:
            self.ok_streak = 0
            self.bad_streak += 1
            if self.bad_streak >= self.cfg.close_after_frames:
                self.open = False
        return {"status": status, "gate_open": self.open,
                "ok_streak": self.ok_streak, "bad_streak": self.bad_streak}

    def reset(self):
        self.open = False
        self.ok_streak = 0
        self.bad_streak = 0
