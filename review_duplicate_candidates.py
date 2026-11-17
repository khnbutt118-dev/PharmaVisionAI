#!/usr/bin/env python3
"""
Create a browser-based visual review gallery for near-duplicate image pairs.

Run from the inner PharmaSight-AI project folder:
    python .\review_duplicate_candidates.py

The script does not modify images, the manifest, or the duplicate report.
It creates:
    reports\dataset_audit\duplicate_review_gallery.html
    reports\dataset_audit\duplicate_review_template.csv
"""

from __future__ import annotations

import csv
import html
from pathlib import Path
from urllib.parse import quote

PROJECT_ROOT = Path(__file__).resolve().parent
DUPLICATE_CSV = PROJECT_ROOT / "reports" / "dataset_audit" / "duplicate_candidates.csv"
MANIFEST_CSV = PROJECT_ROOT / "datasets" / "dataset_manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "reports" / "dataset_audit"
GALLERY_HTML = OUTPUT_DIR / "duplicate_review_gallery.html"
REVIEW_CSV = OUTPUT_DIR / "duplicate_review_template.csv"


def resolve_image_path(raw_path: str) -> Path | None:
    """Resolve paths stored as absolute paths or relative to project root."""
    if not raw_path:
        return None

    p = Path(raw_path)
    candidates = [p] if p.is_absolute() else [PROJECT_ROOT / p, p]

    # Some manifests store paths beginning at the dataset directory.
    normalized = raw_path.replace("\\", "/")
    marker = "datasets/"
    if marker in normalized:
        suffix = normalized.split(marker, 1)[1]
        candidates.append(PROJECT_ROOT / "datasets" / Path(suffix))

    for candidate in candidates:
        try:
            if candidate.is_file():
                return candidate.resolve()
        except OSError:
            pass
    return None


def file_uri(path: Path) -> str:
    # as_uri safely handles Windows drive letters and spaces.
    return path.resolve().as_uri()


def load_manifest_metadata():
    metadata = {}
    if not MANIFEST_CSV.exists():
        return metadata

    with MANIFEST_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            metadata[row.get("image_path", "")] = {
                "split": row.get("split", "") or "unknown",
                "label": row.get("label", "") or "unknown",
            }
    return metadata


def main():
    if not DUPLICATE_CSV.exists():
        raise SystemExit(
            f"ERROR: Duplicate report not found:\n{DUPLICATE_CSV}\n"
            "Run the duplicate-finding command first."
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = load_manifest_metadata()

    with DUPLICATE_CSV.open("r", newline="", encoding="utf-8-sig") as f:
        pairs = list(csv.DictReader(f))

    if not pairs:
        raise SystemExit("No duplicate candidates were found in the CSV.")

    # Prioritize candidates that cross dataset splits or labels.
    def priority(row):
        a = metadata.get(row.get("image_a", ""), {})
        b = metadata.get(row.get("image_b", ""), {})
        cross_split = a.get("split") != b.get("split")
        cross_label = a.get("label") != b.get("label")
        return (not (cross_split or cross_label), int(row.get("hamming_distance") or 999))

    pairs.sort(key=priority)

    review_rows = []
    cards = []

    for index, row in enumerate(pairs, start=1):
        image_a_raw = row.get("image_a", "")
        image_b_raw = row.get("image_b", "")
        image_a = resolve_image_path(image_a_raw)
        image_b = resolve_image_path(image_b_raw)

        meta_a = metadata.get(image_a_raw, {})
        meta_b = metadata.get(image_b_raw, {})
        split_a = meta_a.get("split", "unknown")
        split_b = meta_b.get("split", "unknown")
        label_a = meta_a.get("label", "unknown")
        label_b = meta_b.get("label", "unknown")
        distance = row.get("hamming_distance", "")
        kind = row.get("kind", "near")

        flags = []
        if split_a != split_b:
            flags.append("CROSSES SPLITS")
        if label_a != label_b:
            flags.append("CROSSES LABELS")
        flag_text = " · ".join(flags) if flags else "Same split/label (check visually)"

        def image_panel(path, raw, label, split, side):
            if path:
                image_html = (
                    f'<img src="{html.escape(file_uri(path), quote=True)}" '
                    f'alt="Candidate {side}" loading="lazy">'
                )
                status = html.escape(str(path))
            else:
                image_html = '<div class="missing">IMAGE FILE NOT FOUND</div>'
                status = html.escape(raw)
            return f"""
              <div class="image-panel">
                <div class="image-meta"><b>{side}</b> · {html.escape(split)} / {html.escape(label)}</div>
                {image_html}
                <div class="path">{status}</div>
              </div>
            """

        cards.append(f"""
        <section class="pair">
          <div class="pair-head">
            <b>Pair {index}</b>
            <span>{html.escape(str(kind))} · Hamming distance: {html.escape(str(distance))}</span>
            <strong>{html.escape(flag_text)}</strong>
          </div>
          <div class="images">
            {image_panel(image_a, image_a_raw, label_a, split_a, "Image A")}
            {image_panel(image_b, image_b_raw, label_b, split_b, "Image B")}
          </div>
          <div class="review">
            <label for="status-{index}">Your decision:</label>
            <select id="status-{index}" data-pair="{index}">
              <option value="not_reviewed" selected>Not reviewed</option>
              <option value="same_physical_image">Same physical image / duplicate</option>
              <option value="different_images">Different images</option>
              <option value="uncertain">Uncertain — needs expert review</option>
              <option value="possible_label_error">Possible label error</option>
            </select>
          </div>
        </section>
        """)

        review_rows.append({
            "pair_number": index,
            "image_a": image_a_raw,
            "image_b": image_b_raw,
            "kind": kind,
            "hamming_distance": distance,
            "split_a": split_a,
            "label_a": label_a,
            "split_b": split_b,
            "label_b": label_b,
            "review_decision": "not_reviewed",
        })

    with REVIEW_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(review_rows[0].keys()))
        writer.writeheader()
        writer.writerows(review_rows)

    page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PharmaSight AI — Duplicate Candidate Review</title>
<style>
  body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; color: #202124; background: #f7f8fa; }}
  h1 {{ margin-bottom: 6px; }}
  .note {{ background: #fff8df; border: 1px solid #ead38b; padding: 12px 16px; border-radius: 8px; max-width: 1000px; }}
  .toolbar {{ position: sticky; top: 0; background: #f7f8fa; padding: 12px 0; z-index: 2; }}
  .pair {{ background: white; border: 1px solid #ddd; border-radius: 10px; padding: 14px; margin: 16px 0; }}
  .pair-head {{ display: flex; flex-wrap: wrap; gap: 12px; align-items: center; margin-bottom: 12px; }}
  .pair-head strong {{ color: #9a4d00; }}
  .images {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; }}
  .image-panel {{ min-width: 0; }}
  .image-meta {{ margin-bottom: 6px; }}
  img {{ width: 100%; height: 300px; object-fit: contain; background: #eee; border: 1px solid #ddd; }}
  .path {{ font-size: 11px; color: #666; overflow-wrap: anywhere; margin-top: 5px; }}
  .missing {{ height: 300px; display: grid; place-items: center; background: #fee; color: #900; }}
  .review {{ margin-top: 12px; display: flex; gap: 10px; align-items: center; }}
  select, button {{ font-size: 14px; padding: 8px; }}
  button {{ cursor: pointer; }}
  @media (max-width: 700px) {{ .images {{ grid-template-columns: 1fr; }} img, .missing {{ height: 230px; }} }}
</style>
</head>
<body>
<h1>PharmaSight AI — Near-Duplicate Review</h1>
<p>Candidate pairs: <b>{len(pairs)}</b>. Review images side by side. Similar-looking tablets are not automatically duplicates.</p>
<div class="note">
  <b>Important:</b> Do not delete, move, relabel, or merge any image based only on this report.
  A near-duplicate algorithm can flag visually similar but genuinely different tablets.
  In particular, investigate pairs marked as crossing splits or labels.
</div>
<div class="toolbar">
  <button onclick="downloadDecisions()">Download review decisions CSV</button>
  <span id="progress"></span>
</div>
{''.join(cards)}
<script>
function downloadDecisions() {{
  const rows = [["pair_number","review_decision"]];
  document.querySelectorAll("select[data-pair]").forEach(el => {{
    rows.push([el.dataset.pair, el.value]);
  }});
  const csv = rows.map(row => row.map(v => '"' + String(v).replaceAll('"','""') + '"').join(",")).join("\\r\\n");
  const blob = new Blob([csv], {{type: "text/csv;charset=utf-8"}});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = "duplicate_review_decisions.csv";
  a.click();
  URL.revokeObjectURL(url);
}}
function updateProgress() {{
  const selects = [...document.querySelectorAll("select[data-pair]")];
  const done = selects.filter(s => s.value !== "not_reviewed").length;
  document.getElementById("progress").textContent = `${{done}} / ${{selects.length}} reviewed`;
}}
document.querySelectorAll("select").forEach(s => s.addEventListener("change", updateProgress));
updateProgress();
</script>
</body>
</html>
"""
    GALLERY_HTML.write_text(page, encoding="utf-8")

    print("SUCCESS: Duplicate review gallery created.")
    print(f"Candidate pairs: {len(pairs)}")
    print(f"Gallery: {GALLERY_HTML}")
    print(f"Review template: {REVIEW_CSV}")
    print("Open the HTML file in Edge or Chrome. The script did not modify source images or the manifest.")


if __name__ == "__main__":
    main()
