import argparse
from collections import Counter, defaultdict
from pathlib import Path
import csv

#counts the total filled, and unfilled rows for labeling a CSV file
def analyze_csv(path: Path):
    total = 0
    filled = 0
    empty = 0
    per_image = defaultdict(lambda: {"total": 0, "filled": 0, "empty": 0})
    #counts number of each label
    labels_counter = Counter()

    with path.open(newline="", encoding="utf-8", errors="ignore") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        has_header = False
        if header and header[:3] == ["filename", "person_index", "label"]:
            has_header = True
        else:
            # If first row isn't header, treat it as data
            if header:
                filename, person_index, label = (header + [""] * 3)[:3]
                label = label.strip()
                total += 1
                per_image[filename]["total"] += 1
                if label:
                    filled += 1
                    per_image[filename]["filled"] += 1
                    labels_counter[label] += 1
                else:
                    empty += 1
                    per_image[filename]["empty"] += 1

        for row in reader:
            if not row:
                continue
            filename, person_index, label = (row + [""] * 3)[:3]
            label = label.strip()
            total += 1
            per_image[filename]["total"] += 1
            if label:
                filled += 1
                per_image[filename]["filled"] += 1
                labels_counter[label] += 1
            else:
                empty += 1
                per_image[filename]["empty"] += 1

    # Find images with most empties
    worst_missing = sorted(per_image.items(), key=lambda kv: (kv[1]["empty"], -kv[1]["filled"]), reverse=True)[:10]

    return {
        "path": str(path),
        "total": total,
        "filled": filled,
        "empty": empty,
        "labels": labels_counter,
        "worst_missing": worst_missing,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze label coverage in CSV files")
    parser.add_argument("files", nargs="*", default=[
        "dataset/labels_posture.csv",
        "dataset/labels_gesture.csv",
        "dataset/labels_engagement.csv",
    ], help="CSV files to analyze")
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    for rel in args.files:
        path = (root / rel).resolve()
        if not path.exists():
            print(f"Missing file: {path}")
            continue

        stats = analyze_csv(path)

        # Print summary stats
        print(f"\n[{stats['path']}]")
        print(f"rows: {stats['total']} | filled: {stats['filled']} | empty: {stats['empty']}")
        if stats['total']:
            pct = 100.0 * stats['filled'] / stats['total']
            print(f"coverage: {pct:.1f}%")
        # Top 8 labels
        common = stats['labels'].most_common(8)
        if common:
            print("top labels:")
            for lab, cnt in common:
                print(f"  {lab}: {cnt}")
        # Images with most empties
        if stats['worst_missing']:
            print("most missing (filename: empty/total):")
            for fname, counts in stats['worst_missing']:
                print(f"  {fname}: {counts['empty']}/{counts['total']}")


if __name__ == "__main__":
    main()
