"""Suggest labels from folder/file names (review aid only).

The manifest builder leaves labels empty on purpose. Many datasets (MVTec AD, pillQC,
Roboflow exports) encode the class in folder names, so typing thousands of rows by
hand is error-prone. This script proposes `suggested_label` / `suggested_defect_type`
in a SEPARATE file. It never edits the manifest. Check every suggestion against the
dataset's own documentation before copying it into the manifest.
"""
import argparse
import re
from pathlib import Path
import pandas as pd

NORMAL_TOKENS = {"good", "normal", "ok", "pass", "intact", "nodefect", "nodefects"}
GENERIC_DEFECT_TOKENS = {"defect", "defects", "defective", "bad", "anomaly", "anomalous",
                         "ng", "fail", "failed", "notgood", "faulty"}
SUBTYPE_TOKENS = {
    "crack": "cracking", "cracked": "cracking", "cracking": "cracking",
    "broken": "cracking", "fracture": "cracking",
    "chip": "chipping", "chipped": "chipping", "chipping": "chipping",
    "capping": "capping",
    "dirt": "contamination", "dirty": "contamination", "contaminated": "contamination",
    "contamination": "contamination", "stain": "contamination",
}
# Visible defect words whose subtype is not in our taxonomy -> defective, subtype unresolved.
OTHER_DEFECT_TOKENS = {"scratch", "squeeze", "poke", "imprint", "color", "colour"}


def _tokens(parts):
    toks = []
    for part in parts:
        toks.extend(t for t in re.split(r"[^a-z0-9]+", part.lower()) if t)
    return toks


def suggest(rel_parts):
    """Return (label, defect_type, basis) from path parts BELOW the source folder."""
    toks = _tokens(rel_parts)
    defect_hits, normal_hits, subtypes = [], [], set()
    i = 0
    while i < len(toks):
        t = toks[i]
        nxt = toks[i + 1] if i + 1 < len(toks) else ""
        if t in {"not", "non"} and nxt in NORMAL_TOKENS:
            defect_hits.append(f"{t}_{nxt}")
            i += 2
            continue
        if t == "no" and nxt in {"defect", "defects"}:
            normal_hits.append("no_defect")
            i += 2
            continue
        if t in SUBTYPE_TOKENS:
            defect_hits.append(t)
            subtypes.add(SUBTYPE_TOKENS[t])
        elif t in GENERIC_DEFECT_TOKENS or t in OTHER_DEFECT_TOKENS:
            defect_hits.append(t)
        elif t in NORMAL_TOKENS:
            normal_hits.append(t)
        i += 1
    if defect_hits:
        subtype = next(iter(subtypes)) if len(subtypes) == 1 else "unresolved"
        return "defective", subtype, ",".join(defect_hits)
    if normal_hits:
        return "normal", "none", ",".join(normal_hits)
    return "", "", ""


def run(manifest: Path, root: Path) -> pd.DataFrame:
    df = pd.read_csv(manifest).fillna("")
    rows = []
    for _, r in df.iterrows():
        p = Path(r["image_path"])
        try:
            rel = p.relative_to(root)
            parts = list(rel.parts[1:-1]) + [rel.stem]  # drop source folder, keep sub-folders + file stem
        except ValueError:
            parts = [p.stem]
        label, subtype, basis = suggest(parts)
        rows.append({"image_path": r["image_path"], "source": r.get("source", ""),
                     "current_label": r.get("label", ""),
                     "suggested_label": label, "suggested_defect_type": subtype,
                     "basis": basis})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, default=Path("datasets/dataset_manifest.csv"))
    ap.add_argument("--root", type=Path, default=Path("datasets/raw/tablet_defects"))
    ap.add_argument("--output", type=Path, default=Path("datasets/label_suggestions.csv"))
    args = ap.parse_args()
    if not args.manifest.exists():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    out = run(args.manifest, args.root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(out.groupby(["source", "suggested_label"], dropna=False).size().rename("images").to_string())
    print(f"\nUnlabelled (no suggestion): {int((out.suggested_label == '').sum())}")
    print(f"Saved: {args.output}")
    print("These are SUGGESTIONS from folder names. Verify against dataset documentation "
          "before copying into the manifest. Note: MVTec AD 'train' folders contain normal images only.")


if __name__ == "__main__":
    main()
