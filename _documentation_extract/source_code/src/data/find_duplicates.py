"""Find exact and near-duplicate images (candidates for human review).

Exact duplicates: identical file bytes (SHA-256).
Near duplicates: difference-hash (dHash, 64 bit) with a Hamming-distance threshold.
Near-duplicate results are CANDIDATES: plain/uniform backgrounds can cause false matches,
so look at the pairs before removing anything. Pairs that cross source folders are flagged
because they are the most dangerous kind of train/test leakage. Originals are never modified.
"""
import argparse
import hashlib
from pathlib import Path
import cv2
import numpy as np
import pandas as pd

EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def dhash(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    small = cv2.resize(img, (9, 8), interpolation=cv2.INTER_AREA)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_of(path: Path, root: Path):
    try:
        rel = path.relative_to(root)
        return rel.parts[0] if len(rel.parts) > 1 else "unsorted_source"
    except ValueError:
        return "unknown"


def find(root: Path, threshold: int = 5) -> pd.DataFrame:
    paths = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in EXTS)
    sha, hashes = {}, {}
    for p in paths:
        sha[p] = sha256(p)
        h = dhash(p)
        if h is not None:
            hashes[p] = h
    rows = []
    seen_exact = set()
    by_sha = {}
    for p, s in sha.items():
        by_sha.setdefault(s, []).append(p)
    for group in by_sha.values():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                seen_exact.add((a, b))
                rows.append((a, b, "exact", 0))
    keys = list(hashes)
    arr = np.array([hashes[k] for k in keys], dtype=np.uint64)
    for i in range(len(keys)):
        if i + 1 >= len(keys):
            break
        x = np.bitwise_xor(arr[i + 1:], arr[i])
        dist = np.unpackbits(x.view(np.uint8).reshape(-1, 8), axis=1).sum(axis=1)
        for off in np.nonzero(dist <= threshold)[0]:
            j = i + 1 + int(off)
            a, b = keys[i], keys[j]
            if (a, b) not in seen_exact:
                rows.append((a, b, "near", int(dist[off])))
    out = pd.DataFrame([{
        "image_a": str(a), "image_b": str(b), "kind": k, "hamming_distance": d,
        "source_a": source_of(a, root), "source_b": source_of(b, root),
        "cross_source": source_of(a, root) != source_of(b, root),
    } for a, b, k, d in rows], columns=["image_a", "image_b", "kind", "hamming_distance",
                                         "source_a", "source_b", "cross_source"])
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--threshold", type=int, default=5, help="max Hamming distance (0-64) for near-duplicates")
    ap.add_argument("--output", type=Path, default=Path("reports/dataset_audit/duplicate_candidates.csv"))
    args = ap.parse_args()
    if not args.root.exists():
        raise SystemExit(f"Folder does not exist: {args.root}")
    df = find(args.root, args.threshold)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Exact duplicate pairs: {int((df.kind == 'exact').sum())}")
    print(f"Near-duplicate candidate pairs: {int((df.kind == 'near').sum())}")
    print(f"Cross-source pairs (leakage risk): {int(df.cross_source.sum())}")
    print(f"Report saved: {args.output}")
    print("Review pairs visually before removing or regrouping anything.")


if __name__ == "__main__":
    main()
