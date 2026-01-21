'''
To annotate the people in the images with their index using YoloPose
Whole folder, save annotated images:
python ./yolo_pose/visualize_indices.py --folder ./dataset/images --save-dir ./dataset/outputs --max-width 900 --max-height 600 --no-show
'''
import argparse
import os
from pathlib import Path
import sys

import cv2
from ultralytics import YOLO


def resolve_path(p: str | None) -> Path | None:
    if not p:
        return None
    cand = []
    pth = Path(p)
    cand.append(pth)
    cand.append(Path.cwd() / pth)
    # project root (this file -> yolo_pose -> project)
    proj_root = Path(__file__).resolve().parent.parent
    cand.append(proj_root / pth)
    # if path redundantly includes the project folder name, strip it
    parts = pth.parts
    if 'nonverbal_cue_system' in parts:
        idx = parts.index('nonverbal_cue_system')
        tail = Path(*parts[idx + 1 :])
        cand.append(proj_root / tail)
    for c in cand:
        if c.exists():
            return c.resolve()
    return (Path.cwd() / pth).resolve()


# scale image for display to fit 1280x720
def scale_for_display(img, max_width=1280, max_height=720):
    h, w = img.shape[:2]
    scale = min(max_width / float(w), max_height / float(h), 1.0)
    if scale < 1.0:
        new_w, new_h = int(w * scale), int(h * scale)
        return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return img


# Adds index annotations to each detected person in the result
def annotate_indices_on_result(result, base_img):
    # result: single Ultralytics result for one image
    try:
        annotated = result.plot()
    except Exception:
        annotated = base_img.copy()

    h, w = annotated.shape[:2]
    # Auto-scale font and thickness relative to image size
    scale = max(0.7, min(2.5, w / 640.0))
    thickness = max(2, int(2 * scale))

    # Prefer anchoring labels to detection boxes if available
    boxes = []
    try:
        if hasattr(result, 'boxes') and result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
    except Exception:
        boxes = []

    keypoints = []
    if hasattr(result, 'keypoints') and result.keypoints is not None:
        try:
            keypoints = result.keypoints.xy.cpu().numpy()
        except Exception:
            keypoints = []

    n = max(len(boxes), len(keypoints))
    for idx in range(n):
        # Compute anchor position
        x_anchor, y_anchor = 10, 30
        if idx < len(boxes):
            x1, y1, x2, y2 = boxes[idx]
            x_anchor, y_anchor = int(x1), int(max(0, y1 - 10))
        elif idx < len(keypoints):
            person = keypoints[idx]
            coords = person[:, :2] if person.ndim == 2 else person
            if coords.shape[0] >= 1:
                x_anchor, y_anchor = int(coords[0][0]), int(coords[0][1])  # nose

        label = f"#{idx}"
        # Draw high-contrast text with background box
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        pad = max(2, int(3 * scale))
        x_bg1 = max(0, x_anchor - pad)
        y_bg1 = max(0, y_anchor - th - pad)
        x_bg2 = min(w - 1, x_anchor + tw + pad)
        y_bg2 = min(h - 1, y_anchor + baseline + pad)
        cv2.rectangle(annotated, (x_bg1, y_bg1), (x_bg2, y_bg2), (0, 0, 0), -1)
        cv2.putText(annotated, label, (x_anchor, y_anchor), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 255), thickness, cv2.LINE_AA)

    return annotated

# Loads an image, run YOLO model for detection, annotate indices, save result
def process_image(model, image_path: Path, max_width=1280, max_height=720, show=True, save_dir: Path | None = None):
    img = cv2.imread(str(image_path))
    if img is None:
        raise FileNotFoundError(f"Image not found or unreadable: {image_path}")
    results = model(img)
    if not results:
        return 0
    r = results[0]
    annotated = annotate_indices_on_result(r, img)

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        out_path = save_dir / f"{image_path.stem}_indexed.jpg"
        try:
            cv2.imwrite(str(out_path), annotated)
        except Exception as e:
            print(f"Failed to save {out_path}: {e}", file=sys.stderr)

    if show:
        disp = scale_for_display(annotated, max_width, max_height)
        cv2.namedWindow("Indices", cv2.WINDOW_NORMAL)
        cv2.imshow("Indices", disp)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # return number of people found
    try:
        return len(r.keypoints.xy)
    except Exception:
        return 0

# Iterates through all images in a folder, calling "process_image" for each.
def process_folder(model, folder: Path, max_width=1280, max_height=720, show=True, save_dir: Path | None = None):
    exts = {".jpg", ".jpeg", ".png"}
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"No images found in {folder}")
        return 0
    total = 0
    for p in files:
        try:
            n = process_image(model, p, max_width, max_height, show, save_dir)
            total += n
        except Exception as e:
            print(f"Error processing {p}: {e}", file=sys.stderr)
    print(f"Annotated {len(files)} images in {folder}")
    return total


def main():
    ap = argparse.ArgumentParser(description="Visualize person indices to help labeling.")
    ap.add_argument("--image", "-i", help="Path to a single image")
    ap.add_argument("--folder", "-f", help="Path to a folder of images")
    ap.add_argument("--weights", default="yolov8x-pose.pt", help="YOLO pose weights")
    ap.add_argument("--save-dir", help="Directory to save annotated images")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--max-height", type=int, default=720)
    ap.add_argument("--no-show", action="store_true")
    args = ap.parse_args()

    weights = resolve_path(args.weights)
    model = YOLO(str(weights))

    show = not args.no_show
    save_dir = resolve_path(args.save_dir) if args.save_dir else None

    if args.folder:
        folder = resolve_path(args.folder)
        if not folder or not folder.is_dir():
            raise FileNotFoundError(f"Folder not found: {args.folder}")
        process_folder(model, folder, args.max_width, args.max_height, show, save_dir)
    elif args.image:
        image = resolve_path(args.image)
        if not image or not image.exists():
            raise FileNotFoundError(f"Image not found: {args.image}")
        process_image(model, image, args.max_width, args.max_height, show, save_dir)
    else:
        # default: try dataset/images from project root
        default_folder = Path(__file__).resolve().parent.parent / "dataset" / "images"
        if default_folder.exists():
            process_folder(model, default_folder, args.max_width, args.max_height, show, save_dir)
        else:
            print("Provide --image or --folder")


if __name__ == "__main__":
    main()
