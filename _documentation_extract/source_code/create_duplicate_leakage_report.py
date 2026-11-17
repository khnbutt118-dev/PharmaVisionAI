import csv
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent

DUPLICATE_CSV = ROOT / "reports" / "dataset_audit" / "duplicate_candidates.csv"
MANIFEST_CSV = ROOT / "datasets" / "dataset_manifest.csv"
REVIEW_CSV = ROOT / "reports" / "dataset_audit" / "duplicate_review_decisions_updated.csv"
OUTPUT_CSV = ROOT / "reports" / "dataset_audit" / "duplicate_leakage_report.csv"
SUMMARY_TXT = ROOT / "reports" / "dataset_audit" / "duplicate_leakage_summary.txt"


def read_csv(path):
    if not path.exists():
        raise SystemExit(f"ERROR: File not found: {path}")

    with path.open("r", newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


def load_manifest_metadata():
    metadata = {}

    for row in read_csv(MANIFEST_CSV):
        metadata[row.get("image_path", "")] = {
            "split": row.get("split", "") or "unknown",
            "label": row.get("label", "") or "unknown",
        }

    return metadata


def main():
    print("Loading dataset files...")

    pairs = read_csv(DUPLICATE_CSV)
    reviews = read_csv(REVIEW_CSV)
    metadata = load_manifest_metadata()

    print("Original candidate pairs:", len(pairs))
    print("Review decisions:", len(reviews))

    if len(pairs) != 227 or len(reviews) != 227:
        raise SystemExit(
            "ERROR: Expected exactly 227 candidate pairs and 227 review decisions."
        )

    # Validate review pair numbers.
    review_map = {}

    for row in reviews:
        try:
            number = int(row["pair_number"])
        except (ValueError, KeyError):
            raise SystemExit("ERROR: Invalid or missing pair_number in review CSV.")

        if number in review_map:
            raise SystemExit(f"ERROR: Duplicate review pair number: {number}")

        review_map[number] = row.get("review_decision", "").strip()

    expected_numbers = set(range(1, 228))

    if set(review_map) != expected_numbers:
        missing = sorted(expected_numbers - set(review_map))
        extra = sorted(set(review_map) - expected_numbers)
        raise SystemExit(
            f"ERROR: Pair numbers do not match.\nMissing: {missing}\nExtra: {extra}"
        )

    allowed = {
        "same_physical_image",
        "different_images",
        "uncertain",
        "possible_label_error",
    }

    invalid = {
        number: decision
        for number, decision in review_map.items()
        if decision not in allowed
    }

    if invalid:
        raise SystemExit(f"ERROR: Invalid review decisions: {invalid}")

    # Reproduce the exact sorting used by review_duplicate_candidates.py.
    def priority(row):
        a = metadata.get(row.get("image_a", ""), {})
        b = metadata.get(row.get("image_b", ""), {})

        cross_split = a.get("split") != b.get("split")
        cross_label = a.get("label") != b.get("label")

        return (
            not (cross_split or cross_label),
            int(row.get("hamming_distance") or 999),
        )

    pairs.sort(key=priority)

    output_rows = []

    for index, row in enumerate(pairs, start=1):
        image_a = row.get("image_a", "")
        image_b = row.get("image_b", "")

        meta_a = metadata.get(image_a, {})
        meta_b = metadata.get(image_b, {})

        split_a = meta_a.get("split", "unknown")
        split_b = meta_b.get("split", "unknown")
        label_a = meta_a.get("label", "unknown")
        label_b = meta_b.get("label", "unknown")

        decision = review_map[index]

        cross_split = split_a != split_b
        cross_label = label_a != label_b

        if decision == "same_physical_image":
            if cross_split:
                action = "REGROUP_REQUIRED: same tablet appears in different splits"
            elif cross_label:
                action = "INVESTIGATE: same tablet has conflicting labels"
            else:
                action = "SAME_SPLIT: retain together; verify sample grouping"

        elif decision == "possible_label_error":
            action = "MANUAL_LABEL_INVESTIGATION"

        elif decision == "uncertain":
            action = "MANUAL_REVIEW_REQUIRED"

        else:
            action = "NO_DUPLICATE_ACTION"

        output_rows.append({
            "pair_number": index,
            "image_a": image_a,
            "image_b": image_b,
            "review_decision": decision,
            "split_a": split_a,
            "split_b": split_b,
            "label_a": label_a,
            "label_b": label_b,
            "cross_split": cross_split,
            "cross_label": cross_label,
            "kind": row.get("kind", ""),
            "hamming_distance": row.get("hamming_distance", ""),
            "source_a": row.get("source_a", ""),
            "source_b": row.get("source_b", ""),
            "recommended_action": action,
        })

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(output_rows[0].keys()))
        writer.writeheader()
        writer.writerows(output_rows)

    # Create summary counts.
    decisions = Counter(r["review_decision"] for r in output_rows)

    same_pairs = [
        r for r in output_rows
        if r["review_decision"] == "same_physical_image"
    ]

    same_cross_split = [
        r for r in same_pairs if r["cross_split"]
    ]

    same_cross_label = [
        r for r in same_pairs if r["cross_label"]
    ]

    all_cross_split = [
        r for r in output_rows if r["cross_split"]
    ]

    all_cross_label = [
        r for r in output_rows if r["cross_label"]
    ]

    summary = [
        "PHARMASIGHT AI - DUPLICATE LEAKAGE AUDIT",
        "=" * 55,
        f"Total candidate pairs: {len(output_rows)}",
        f"Same physical image: {decisions['same_physical_image']}",
        f"Different images: {decisions['different_images']}",
        f"Uncertain: {decisions['uncertain']}",
        f"Possible label errors: {decisions['possible_label_error']}",
        "",
        f"Same-image pairs crossing splits: {len(same_cross_split)}",
        f"Same-image pairs crossing labels: {len(same_cross_label)}",
        f"All pairs crossing splits: {len(all_cross_split)}",
        f"All pairs crossing labels: {len(all_cross_label)}",
        "",
        "SAME-IMAGE PAIRS CROSSING SPLITS:",
    ]

    if same_cross_split:
        for r in same_cross_split:
            summary.append(
                f"Pair {r['pair_number']}: "
                f"{r['split_a']} -> {r['split_b']} | "
                f"{r['image_a']} | {r['image_b']}"
            )
    else:
        summary.append("None found.")

    summary += [
        "",
        "SAME-IMAGE PAIRS CROSSING LABELS:",
    ]

    if same_cross_label:
        for r in same_cross_label:
            summary.append(
                f"Pair {r['pair_number']}: "
                f"{r['label_a']} -> {r['label_b']} | "
                f"{r['image_a']} | {r['image_b']}"
            )
    else:
        summary.append("None found.")

    summary += [
        "",
        "IMPORTANT:",
        "This report does not modify or move images.",
        "Pair matches are based on the gallery's original numbering.",
        "Review physical-sample grouping before changing dataset splits.",
        "Unknown metadata must be resolved before model training.",
    ]

    SUMMARY_TXT.write_text("\n".join(summary), encoding="utf-8")

    print("\nAUDIT COMPLETED")
    print("-" * 45)
    print("Same physical image:", decisions["same_physical_image"])
    print("Different images:", decisions["different_images"])
    print("Uncertain:", decisions["uncertain"])
    print("Possible label errors:", decisions["possible_label_error"])
    print("Same-image pairs crossing splits:", len(same_cross_split))
    print("Same-image pairs crossing labels:", len(same_cross_label))
    print("\nCSV report:", OUTPUT_CSV)
    print("Summary:", SUMMARY_TXT)
    print("\nNo source images or manifest rows were modified.")


if __name__ == "__main__":
    main()
