# Convert YOLO Pose Output to Dataset for ML
# This will be used to build posture, gesture, and engagement_features.csv

'''
python nonverbal_cue_system/yolo_pose/extract_keypoints.py --images nonverbal_cue_system/dataset/images --labels-file nonverbal_cue_system/dataset/labels_posture.csv --task posture --output nonverbal_cue_system/training/posture_features.csv --overwrite
python nonverbal_cue_system/yolo_pose/extract_keypoints.py --images nonverbal_cue_system/dataset/images --labels-file nonverbal_cue_system/dataset/labels_gesture.csv --task gesture --output nonverbal_cue_system/training/gesture_features.csv --overwrite
python nonverbal_cue_system/yolo_pose/extract_keypoints.py --images nonverbal_cue_system/dataset/images --labels-file nonverbal_cue_system/dataset/labels_engagement.csv --task engagement --output nonverbal_cue_system/training/engagement_features.csv --overwrite
'''


import os
import argparse
import pandas as pd
import numpy as np
from ultralytics import YOLO
import cv2
from pathlib import Path


def keypoints_to_features(kp_tensor):
    # Accept tensor-like of shape (N_people, N_keypoints, 2 or 3) and turn it to NumPy array x,y coordinates
    all_people = []

    for kp in kp_tensor:
        arr = kp.cpu().numpy()
        # if third value is confidence, take only x,y
        if arr.ndim == 2 and arr.shape[1] >= 2:
            arr_xy = arr[:, :2]
        else:
            arr_xy = arr

        flat = arr_xy.flatten()
        all_people.append(flat)

    return np.array(all_people)

def compute_posture_features(coords):
    # coords: (N_kp, 2) in image space
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])
    
    # 7 keypoints indices for posture:
    # 0: nose, 5: left shoulder, 6: right shoulder, 11: left hip, 12: right hip, 9: left wrist, 10: right wrist
    ls, rs = get(5), get(6)
    lh, rh = get(11), get(12)
    nose = get(0)
    # Midpoints
    shoulder_mid = np.nanmean(np.vstack([ls, rs]), axis=0)
    hip_mid = np.nanmean(np.vstack([lh, rh]), axis=0)
    # Torso vector (hip->shoulder) and vertical reference
    torso = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])  # up direction (y decreases upward)
    def angle_deg(u, v):
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return np.nan
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return np.nan
        cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return np.degrees(np.arccos(cos))
    torso_tilt = angle_deg(torso, vertical)  # larger -> leaning forward/backward
    # Head direction proxy: nose -> shoulder_mid vector
    head_vec = nose - shoulder_mid
    head_tilt = angle_deg(head_vec, vertical)
    return np.array([torso_tilt, head_tilt])


# calclutate gesture features using hand, wrist and nose
def compute_gesture_features(coords):
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])
    ls, rs = get(5), get(6)
    lw, rw = get(9), get(10)
    nose = get(0)
    # Elevation of wrists relative to shoulders (negative y is up)
    left_raise = (lw[1] - ls[1]) if not np.isnan(lw[1]) and not np.isnan(ls[1]) else np.nan
    right_raise = (rw[1] - rs[1]) if not np.isnan(rw[1]) and not np.isnan(rs[1]) else np.nan
    # Chin touch proxy: wrist close to nose
    def dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.linalg.norm(a - b))
    left_nose_dist = dist(lw, nose)
    right_nose_dist = dist(rw, nose)
    return np.array([left_raise, right_raise, left_nose_dist, right_nose_dist])

# calculate engagemetn features using shoulders, hip, wrist and noose
def compute_engagement_features(coords, phone_boxes=None):
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])

    nose = get(0)
    ls, rs = get(5), get(6)
    lh, rh = get(11), get(12)
    lw, rw = get(9), get(10)

    shoulder_mid = np.nanmean(np.vstack([ls, rs]), axis=0)
    hip_mid = np.nanmean(np.vstack([lh, rh]), axis=0)

    # Scale: shoulder width
    def safe_dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.linalg.norm(a - b))

    shoulder_width = safe_dist(ls, rs)
    scale = shoulder_width if shoulder_width and not np.isnan(shoulder_width) and shoulder_width > 0 else np.nan

    # Head direction
    head_vec = nose - shoulder_mid
    vertical = np.array([0.0, -1.0])
    def angle_deg(u, v):
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return np.nan
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return np.nan
        cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return np.degrees(np.arccos(cos))
    head_tilt = angle_deg(head_vec, vertical)

    # Wrist to nose and wrist to hip distances (normalized)
    lnose = safe_dist(lw, nose)
    rnose = safe_dist(rw, nose)
    lhip = safe_dist(lw, hip_mid)
    rhip = safe_dist(rw, hip_mid)

    def norm(v):
        return v / scale if scale and not np.isnan(scale) and scale > 0 and not np.isnan(v) else np.nan

    lnose_n = norm(lnose)
    rnose_n = norm(rnose)
    lhip_n = norm(lhip)
    rhip_n = norm(rhip)

    # Optional for phone proximity
    def center(box):
        x1, y1, x2, y2 = box
        return np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])
    def min_dist_to_phones(pt):
        if phone_boxes is None or len(phone_boxes) == 0 or np.any(np.isnan(pt)):
            return np.nan
        centers = [center(b) for b in phone_boxes]
        dists = [np.linalg.norm(pt - c) for c in centers]
        return float(np.min(dists)) if dists else np.nan
    lw_phone = min_dist_to_phones(lw)
    rw_phone = min_dist_to_phones(rw)
    lw_phone_n = norm(lw_phone) if lw_phone is not np.nan else np.nan
    rw_phone_n = norm(rw_phone) if rw_phone is not np.nan else np.nan

    # Relative wrist height (below shoulder implies laptop/desk usage)
    left_wrist_rel = (lw[1] - shoulder_mid[1]) if not np.isnan(lw[1]) and not np.isnan(shoulder_mid[1]) else np.nan
    right_wrist_rel = (rw[1] - shoulder_mid[1]) if not np.isnan(rw[1]) and not np.isnan(shoulder_mid[1]) else np.nan
    left_wrist_rel_n = norm(abs(left_wrist_rel))
    right_wrist_rel_n = norm(abs(right_wrist_rel))

    return np.array([
        head_vec[0] if not np.isnan(head_vec[0]) else np.nan,
        head_vec[1] if not np.isnan(head_vec[1]) else np.nan,
        head_tilt,
        lnose_n, rnose_n,
        lhip_n, rhip_n,
        left_wrist_rel_n, right_wrist_rel_n,
        lw_phone_n, rw_phone_n
    ])

# convests the lists to a DataFrame and saves to CSV
def save_features(rows, output_csv, overwrite=False):
    df = pd.DataFrame(rows)
    if overwrite:
        df.to_csv(output_csv, mode='w', header=False, index=False)
    else:
        header = False
        df.to_csv(output_csv, mode='a', header=header, index=False)

def resolve_path(input_path):
    if not input_path:
        return None
    p = Path(input_path)
    candidates = [p, Path.cwd() / p, Path(__file__).resolve().parent.parent / p]
    # If the provided path includes the project root segment, strip it and try relative to script root
    parts = p.parts
    if 'nonverbal_cue_system' in parts:
        idx = parts.index('nonverbal_cue_system')
        tail = Path(*parts[idx+1:])
        candidates.append(Path(__file__).resolve().parent.parent / tail)
    for cand in candidates:
        if cand.exists():
            return str(cand.resolve())
    return str(p)

# load model, read labels, iterate images, extract features, save to CSV
def run_extraction(images_dir, labels_file=None, labels_dir=None, output_csv='pose_features.csv', model_weights='yolov8x-pose.pt', task='combined', overwrite=False):
    images_dir = resolve_path(images_dir)
    labels_file = resolve_path(labels_file) if labels_file else None
    labels_dir = resolve_path(labels_dir) if labels_dir else None
    output_csv = resolve_path(output_csv)
    model_weights = resolve_path(model_weights)

    if images_dir is None:
        raise FileNotFoundError("Images directory path is missing")
    if output_csv is None:
        raise FileNotFoundError("Output CSV path is missing")
    if model_weights is None:
        raise FileNotFoundError("Model weights path is missing")
    print(f"[extract] images_dir -> {images_dir}")
    print(f"[extract] labels_file -> {labels_file}")
    print(f"[extract] labels_dir -> {labels_dir}")
    print(f"[extract] output_csv -> {output_csv}")
    print(f"[extract] weights -> {model_weights}")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    out_dirname = os.path.dirname(output_csv)
    if out_dirname:
        os.makedirs(out_dirname, exist_ok=True)
    model = YOLO(model_weights)

    # load labels mapping from CSV if provided
    labels_map = {}
    indexed_labels = {}  # filename -> {index: label}
    if labels_file and os.path.exists(labels_file):
        # Allow commented header lines starting with '#', and tolerate header row without '#'
        df_labels = pd.read_csv(labels_file, header=None, comment='#')
        # Support two formats:
        # (a) filename,label
        # (b) filename,person_index,label
        for _, row in df_labels.iterrows():
            # Normalize possible header values to skip
            vals = [str(v).strip() for v in row.tolist()]
            low = [v.lower() for v in vals]
            if len(low) >= 3 and (low[0] == 'filename' or low[1] in ('person_index', 'index') or low[2] == 'label'):
                continue  # skip header-like row
            if len(low) >= 2 and (low[0] == 'filename' or low[1] == 'label'):
                continue  # skip header-like row

            if len(row) >= 3:
                lfname = vals[0]
                sidx = vals[1]
                if sidx.isdigit():
                    idx = int(sidx)
                    lbl = vals[2]
                    indexed_labels.setdefault(lfname, {})[idx] = lbl
                else:
                    # Not a numeric index; treat as filename,label
                    labels_map[vals[0]] = vals[1]
            elif len(row) >= 2:
                labels_map[vals[0]] = vals[1]

    rows = []

    for fname in sorted(os.listdir(images_dir)):
        if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        img_path = os.path.join(images_dir, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: cannot read {img_path}")
            continue

        results = model(img)

        # Try to find any label for this image. If per-person labels exist, do NOT skip.
        base = os.path.splitext(fname)[0]
        image_label = None
        if labels_file and base in labels_map:
            image_label = labels_map[base]
        elif labels_dir:
            lbl_path = os.path.join(labels_dir, base + '.txt')
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    image_label = f.read().strip()

        has_indexed = (fname in indexed_labels and len(indexed_labels[fname]) > 0)
        if (image_label is None) and (not has_indexed) and (not labels_dir):
            # No image-level label, no per-person label, and no labels directory
            print(f"Skipping {fname}: no label found")
            continue

        for r in results:
            if not hasattr(r, 'keypoints') or r.keypoints is None:
                continue

            kp_tensor = r.keypoints.xy  # tensor shape (N_people, N_kp, 2 or 3)
            features = keypoints_to_features(kp_tensor)

            for person_idx, f in enumerate(features):
                # f is flattened coords; reshape to (N_kp, 2)
                if f.size % 2 != 0 or f.size == 0:
                    continue
                coords = f.reshape(-1, 2)
                if task == 'posture':
                    feats = compute_posture_features(coords)
                elif task == 'gesture':
                    feats = compute_gesture_features(coords)
                elif task == 'engagement':
                    feats = compute_engagement_features(coords)
                else:
                    # combined: concat all
                    feats = np.concatenate([
                        compute_posture_features(coords),
                        compute_gesture_features(coords),
                        compute_engagement_features(coords)
                    ])
                # decide label: prefer indexed if provided
                lbl = None
                base = os.path.splitext(fname)[0]
                if fname in indexed_labels and person_idx in indexed_labels[fname]:
                    lbl = indexed_labels[fname][person_idx]
                elif labels_file and base in labels_map:
                    lbl = labels_map[base]
                elif labels_dir:
                    lbl_path = os.path.join(labels_dir, base + '.txt')
                    if os.path.exists(lbl_path):
                        with open(lbl_path, 'r') as ftxt:
                            lines = [ln.strip() for ln in ftxt.readlines() if ln.strip()]
                            if person_idx < len(lines):
                                lbl = lines[person_idx]
                if lbl is None:
                    print(f"Skipping {fname} person {person_idx}: no label found")
                    continue
                row = list(feats) + [lbl]
                rows.append(row)

    if rows:
        save_features(rows, output_csv, overwrite=overwrite)
        print(f"Saved {len(rows)} person-feature rows to {output_csv}")
    else:
        print("No features extracted.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--images', required=True, help='Path to dataset images directory')
    parser.add_argument('--labels-file', help='CSV file with filename,label columns')
    parser.add_argument('--labels-dir', help='Directory with per-image label txt files (name.txt)')
    parser.add_argument('--output', default='pose_features.csv')
    parser.add_argument('--weights', default='yolov8x-pose.pt')
    parser.add_argument('--task', choices=['posture','gesture','engagement','combined'], default='combined')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite output CSV instead of appending')

    args = parser.parse_args()

    run_extraction(args.images, labels_file=args.labels_file, labels_dir=args.labels_dir, output_csv=args.output, model_weights=args.weights, task=args.task, overwrite=args.overwrite)
