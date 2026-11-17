import numpy as np
import cv2
from pathlib import Path
import pandas as pd
from src.data.suggest_labels import suggest, run
from src.data.find_duplicates import find
from src.modules.qa_engine import interpret_probability


def test_suggest_basic_labels():
    assert suggest(["test", "crack", "000"])[:2] == ("defective", "cracking")
    assert suggest(["train", "good", "001"])[:2] == ("normal", "none")
    assert suggest(["not_good", "a"])[0] == "defective"
    assert suggest(["no_defect", "a"])[0] == "normal"
    assert suggest(["misc", "img12"])[0] == ""
    assert suggest(["scratch", "x"])[:2] == ("defective", "unresolved")
    assert suggest(["crack", "chip", "x"])[:2] == ("defective", "unresolved")


def test_source_folder_name_is_ignored(tmp_path):
    # a source called 'pills_defect' must not turn every image into 'defective'
    root = tmp_path / "raw"
    f = root / "pills_defect" / "good" / "a.png"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"x")
    m = tmp_path / "m.csv"
    pd.DataFrame({"image_path": [str(f)], "source": ["pills_defect"], "label": [""]}).to_csv(m, index=False)
    out = run(m, root)
    assert out.loc[0, "suggested_label"] == "normal"


def _img(path, seed):
    rng = np.random.default_rng(seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), (rng.random((64, 64, 3)) * 255).astype("uint8"))


def test_duplicates_exact_and_cross_source(tmp_path):
    _img(tmp_path / "a" / "1.png", 1)
    (tmp_path / "b").mkdir()
    (tmp_path / "b" / "copy.png").write_bytes((tmp_path / "a" / "1.png").read_bytes())
    _img(tmp_path / "b" / "other.png", 2)
    df = find(tmp_path, threshold=5)
    exact = df[df.kind == "exact"]
    assert len(exact) == 1 and bool(exact.iloc[0].cross_source)


def test_interpret_probability_states():
    assert interpret_probability(0.95)["status"].startswith("ATTENTION")
    assert interpret_probability(0.05)["status"].startswith("NO VISIBLE")
    assert interpret_probability(0.50)["status"].startswith("UNCERTAIN")
