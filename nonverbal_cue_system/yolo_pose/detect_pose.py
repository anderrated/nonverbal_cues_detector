# Detect all students in crowded room
# Draw skeletons
# Return keypoint coordinates

from ultralytics import YOLO
import cv2
import argparse
import sys
from pathlib import Path
import os

# Load YOLOv8 pose model
model = YOLO("yolov8x-pose.pt")

def run_pose(image_path, max_width=1280, max_height=720, save_path=None, show=True):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    results = model(img)

    for r in results:
        keypoints = r.keypoints.xy  # list of (17 keypoints × N persons)

        # Draw results
        annotated = r.plot()

        h, w = annotated.shape[:2]
        scale = min(max_width / float(w), max_height / float(h), 1.0)
        display = annotated
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(annotated, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Optional save (writes annotated full-res image)
        if save_path:
            out_p = Path(save_path)
            if out_p.is_dir():
                out_p = out_p / (Path(image_path).stem + "_pose.jpg")
            try:
                cv2.imwrite(str(out_p), annotated)
            except Exception as e:
                print(f"Failed to save {out_p}: {e}", file=sys.stderr)

        if show:
            cv2.namedWindow("YOLO-Pose Output", cv2.WINDOW_NORMAL)
            cv2.imshow("YOLO-Pose Output", display)
            cv2.waitKey(0)

    return results

def run_folder(folder_path, max_width=1280, max_height=720, save_dir=None):
    folder = Path(folder_path)
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

    exts = (".jpg", ".jpeg", ".png")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"No images found in {folder}")
        return 0

    count = 0
    for p in sorted(files):
        out_path = None
        if save_dir:
            out_path = save_dir / (p.stem + "_pose.jpg")
        try:
            run_pose(p, max_width=max_width, max_height=max_height, save_path=out_path)
            count += 1
        except Exception as e:
            print(f"Error processing {p}: {e}", file=sys.stderr)
    print(f"Processed {count} images from {folder}")
    return count

def _find_default_image():
    script_dir = Path(__file__).resolve().parent
    dataset_images = script_dir.parent / "dataset" / "images"
    for pattern in ("*.jpg", "*.jpeg", "*.png"):
        matches = list(dataset_images.glob(pattern))
        if matches:
            return matches[0]
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run YOLOv8 pose estimation on an image.")
    parser.add_argument("--image", "-i", help="Path to input image file")
    parser.add_argument("--folder", "-f", help="Process all images in a folder")
    parser.add_argument("--max-width", type=int, default=1280, help="Max display width")
    parser.add_argument("--max-height", type=int, default=720, help="Max display height")
    parser.add_argument("--save", "-o", help="Optional output path or directory to save annotated image")
    parser.add_argument("--no-show", action="store_true", help="Do not show image windows (batch mode)")
    args = parser.parse_args()

    if args.folder:
        folder = Path(args.folder)
        folder = folder if folder.is_absolute() else (Path.cwd() / folder).resolve()
        run_folder(folder, max_width=args.max_width, max_height=args.max_height, save_dir=args.save)
    else:
        img_path = None
        if args.image:
            p = Path(args.image)
            img_path = p if p.is_absolute() else (Path.cwd() / p).resolve()
        else:
            img_path = _find_default_image()

        if not img_path or not Path(img_path).exists():
            print("Image not found. Provide an image with --image or place one under dataset/images.", file=sys.stderr)
            print(f"Tried path: {img_path}", file=sys.stderr)
            print(f"CWD: {Path.cwd()}", file=sys.stderr)
            sys.exit(1)

        run_pose(img_path, max_width=args.max_width, max_height=args.max_height, save_path=args.save, show=not args.no_show)
