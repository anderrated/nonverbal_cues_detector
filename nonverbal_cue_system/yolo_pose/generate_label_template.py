import os
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO
import cv2


def iter_images(images_dir):
    exts = {".jpg", ".jpeg", ".png"}
    for p in sorted(Path(images_dir).iterdir()):
        if p.is_file() and p.suffix.lower() in exts:
            yield p.name, str(p)


# Parses existing CSV to avoid duplicate entries
def load_existing_pairs(csv_path):
    pairs = set()
    if not os.path.exists(csv_path):
        return pairs
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            parts = [x.strip() for x in s.split(",")]
            if len(parts) >= 2:
                fname = parts[0]
                try:
                    idx = int(parts[1]) if len(parts) >= 3 else None
                except ValueError:
                    continue
                if idx is not None:
                    pairs.add((fname, idx))
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Generate a per-person labeling CSV template by scanning images with YOLO-Pose.")
    parser.add_argument("--images", default=str(Path(__file__).resolve().parent.parent / "dataset" / "images"), help="Images directory")
    parser.add_argument("--output", default=str(Path(__file__).resolve().parent.parent / "dataset" / "labels_gesture.csv"), help="Output CSV path")
    parser.add_argument("--weights", default="yolov8x-pose.pt", help="YOLO pose weights")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing CSV instead of appending missing rows")
    args = parser.parse_args()

    images_dir = Path(args.images)
    if not images_dir.exists() or not images_dir.is_dir():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)

    #load existing pairs to avoid duplicates
    existing = set()
    if out_path.exists() and not args.overwrite:
        existing = load_existing_pairs(str(out_path))

    mode = "w" if args.overwrite else ("a" if out_path.exists() else "w")

    wrote_header = False
    if not out_path.exists() or args.overwrite:
        wrote_header = True

    with open(out_path, mode, newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if wrote_header:
            writer.writerow(["# filename,person_index,label"])
        for fname, fpath in iter_images(images_dir):
            results = model(fpath)
            if not results:
                continue
            # take first result
            r = results[0]
            n_people = 0
            if hasattr(r, "keypoints") and r.keypoints is not None:
                try:
                    n_people = len(r.keypoints.xy)
                except Exception:
                    n_people = 0
            for i in range(n_people):
                if (fname, i) in existing:
                    continue
                # Write placeholder label to be filled by the user
                writer.writerow([fname, i, ""])  # empty third column
    print(f"Template updated at: {out_path}")


if __name__ == "__main__":
    main()
