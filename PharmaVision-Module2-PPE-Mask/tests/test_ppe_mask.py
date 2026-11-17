import json
from pathlib import Path
import numpy as np
import cv2
from src.ppe_mask.gate import (Detection, GateConfig, MaskGate, frame_status, COMPLIANT, NOT_COMPLIANT,
                               UNCERTAIN, NO_FACE)
from src.ppe_mask.prepare_mask_dataset import convert, to_yolo
from src.ppe_mask.evaluate_gate import evaluate
from src.ppe_mask.collect_webcam_faces import yolo_line
from src.ppe_mask.mask_module import draw_overlay, banner_for

BIG = (100, 100, 300, 300)
AREA = 640.0 * 480.0


def D(cls, conf, box=BIG):
    return Detection(cls, conf, box)


def test_frame_status_rules():
    cfg = GateConfig()
    assert frame_status([], cfg, AREA) == NO_FACE
    assert frame_status([D("with_mask", 0.9)], cfg, AREA) == COMPLIANT
    assert frame_status([D("with_mask", 0.5)], cfg, AREA) == UNCERTAIN          # not confident enough
    assert frame_status([D("without_mask", 0.40)], cfg, AREA) == NOT_COMPLIANT  # low conf still blocks
    assert frame_status([D("mask_weared_incorrect", 0.5)], cfg, AREA) == NOT_COMPLIANT
    # one good + one bad face -> blocked
    assert frame_status([D("with_mask", 0.95), D("without_mask", 0.6, (0, 0, 150, 150))], cfg, AREA) == NOT_COMPLIANT
    # tiny background face is ignored
    tiny = D("without_mask", 0.9, (0, 0, 10, 10))
    assert frame_status([D("with_mask", 0.9), tiny], cfg, AREA) == COMPLIANT


def test_low_conf_bad_detection_is_never_a_pass():
    cfg = GateConfig()
    assert frame_status([D("without_mask", 0.2)], cfg, AREA) == UNCERTAIN


def test_gate_open_close():
    g = MaskGate(GateConfig(open_after_frames=3, close_after_frames=2))
    for _ in range(2):
        assert not g.update(COMPLIANT)["gate_open"]
    assert g.update(COMPLIANT)["gate_open"]
    assert g.update(NOT_COMPLIANT)["gate_open"]       # 1 bad frame: still open
    assert not g.update(NOT_COMPLIANT)["gate_open"]   # 2nd bad frame: closes
    assert not g.update(COMPLIANT)["gate_open"]       # must re-earn the streak


def test_to_yolo_and_label_line():
    assert to_yolo((0, 0, 100, 50), 200, 100) == (0.25, 0.25, 0.5, 0.5)
    assert to_yolo((10, 10, 10.5, 40), 200, 100) is None
    assert yolo_line(1, 10, 20, 100, 50, 200, 100) == "1 0.300000 0.450000 0.500000 0.500000"


def _make_voc(raw: Path, name: str, w=120, h=80, objs=(("with_mask", 10, 10, 60, 60),), seed=0, copy_of=None):
    (raw / "images").mkdir(parents=True, exist_ok=True)
    (raw / "annotations").mkdir(parents=True, exist_ok=True)
    p = raw / "images" / f"{name}.png"
    if copy_of:
        p.write_bytes((raw / "images" / f"{copy_of}.png").read_bytes())
    else:
        rng = np.random.default_rng(seed)
        cv2.imwrite(str(p), (rng.random((h, w, 3)) * 255).astype("uint8"))
    body = "".join(f"<object><name>{n}</name><bndbox><xmin>{a}</xmin><ymin>{b}</ymin><xmax>{c}</xmax><ymax>{d}</ymax></bndbox></object>"
                   for n, a, b, c, d in objs)
    (raw / "annotations" / f"{name}.xml").write_text(f"<annotation><filename>{name}.png</filename>{body}</annotation>")


def test_convert_dataset(tmp_path):
    raw = tmp_path / "raw"
    for i in range(20):
        cls = ["with_mask", "without_mask", "mask_weared_incorrect"][i % 3]
        _make_voc(raw, f"img{i}", objs=((cls, 10, 10, 60, 60), ("with_mask", 70, 5, 110, 45)), seed=i)
    _make_voc(raw, "dup_of_0", copy_of="img0", objs=(("with_mask", 10, 10, 60, 60),))
    _make_voc(raw, "badclass", objs=(("banana", 10, 10, 60, 60),), seed=99)
    out = tmp_path / "prep"
    rep = convert(raw, out, seed=1, val_frac=0.2, test_frac=0.2)
    assert rep["duplicate_images_dropped"] == 1
    assert rep["images_used"] == 20
    assert rep["skipped_unknown_class_objects"] == 1 and rep["images_without_objects"] == 1
    sc = rep["split_counts"]
    assert sum(sc[s]["images"] for s in sc) == 20 and sc["test"]["images"] == 4 and sc["val"]["images"] == 4
    names = {s: {p.stem for p in (out / "images" / s).iterdir()} for s in ("train", "val", "test")}
    assert not (names["train"] & names["val"]) and not (names["train"] & names["test"]) and not (names["val"] & names["test"])
    for s in names:
        for stem in names[s]:
            for line in (out / "labels" / s / f"{stem}.txt").read_text().split("\n"):
                if line:
                    c, *v = line.split()
                    assert int(c) in (0, 1, 2) and all(0 <= float(x) <= 1 for x in v)
    assert "names" in (out / "data.yaml").read_text()
    # deterministic
    rep2 = convert(raw, tmp_path / "prep2", seed=1, val_frac=0.2, test_frac=0.2)
    assert rep2["split_counts"] == rep["split_counts"]


def test_webcam_sessions_go_to_train_only(tmp_path):
    raw = tmp_path / "raw"
    for i in range(10):
        _make_voc(raw, f"k{i}", seed=i)
    wc = tmp_path / "webcam" / "s1"
    (wc / "images").mkdir(parents=True)
    (wc / "labels").mkdir(parents=True)
    cv2.imwrite(str(wc / "images" / "a.jpg"), np.zeros((50, 50, 3), "uint8"))
    (wc / "labels" / "a.txt").write_text("1 0.5 0.5 0.4 0.4\n")
    rep = convert(raw, tmp_path / "p", webcam_root=tmp_path / "webcam", train_sessions=["s1"], val_frac=0.2, test_frac=0.2)
    assert rep["webcam_images_added_to_train"] == 1
    assert (tmp_path / "p" / "images" / "train" / "webcam_s1_a.jpg").exists()
    assert not list((tmp_path / "p" / "images" / "test").glob("webcam_*"))


def test_gate_evaluation_counts(tmp_path):
    imgs, labs = tmp_path / "i", tmp_path / "l"
    imgs.mkdir(); labs.mkdir()
    truth = {"a": 0, "b": 0, "c": 1, "d": 1, "e": 2}  # class id per image
    for n, c in truth.items():
        cv2.imwrite(str(imgs / f"{n}.png"), np.zeros((10, 10, 3), "uint8"))
        (labs / f"{n}.txt").write_text(f"{c} 0.5 0.5 0.5 0.5\n")
    preds = {"a": [D("with_mask", 0.9)], "b": [D("with_mask", 0.4)],      # b: truth ok, predicted uncertain -> false reject
             "c": [D("without_mask", 0.8)], "d": [D("with_mask", 0.95)],  # d: false accept
             "e": []}                                                      # e: no face -> not compliant (correct)
    summary, rows = evaluate(imgs, labs, lambda p: (preds[p.stem], AREA))
    assert summary["false_accepts"] == 1 and summary["false_rejects"] == 1
    assert summary["truth_compliant"] == 2 and summary["truth_not_compliant"] == 3
    assert abs(summary["false_accept_rate"] - 1 / 3) < 1e-9


def test_overlay_draws_without_model():
    frame = np.zeros((480, 640, 3), np.uint8)
    dets = [D("without_mask", 0.9), D("with_mask", 0.8, (400, 100, 500, 220))]
    info = {"status": NOT_COMPLIANT, "gate_open": False, "has_no_mask": True, "has_incorrect": False}
    out = draw_overlay(frame.copy(), dets, info)
    assert out.shape == frame.shape and out.any()
    assert "NO MASK" in banner_for(info)[0]
