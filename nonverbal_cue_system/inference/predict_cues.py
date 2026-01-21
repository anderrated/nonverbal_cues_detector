# Detect non-verbal cues
# Detect person, gesture, write label on img

''''
Image:
python nonverbal_cue_system/inference/predict_cues.py --image nonverbal_cue_system/dataset/test_images/test1.jpg --max-width 1600 --max-height 900 --save nonverbal_cue_system/dataset/outputs/test_images --no-show

Test Images Folder:
python nonverbal_cue_system/inference/predict_cues.py --folder nonverbal_cue_system/dataset/test_images --max-width 1600 --max-height 900 --save nonverbal_cue_system/dataset/outputs/test_images --no-show

Webcam:
python nonverbal_cue_system/inference/predict_cues.py --webcam --max-width 1280 --max-height 720

'''

from ultralytics import YOLO
import cv2
import numpy as np
import pickle
import os
import argparse
from pathlib import Path

# Base path for inference assets (models, encoders)
HERE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
INFER_DIR = os.path.join(HERE, 'inference')
INFER_PATH = Path(INFER_DIR)

# initialize YOLO pose model
pose_model = YOLO(os.path.join(os.path.dirname(__file__), '..', 'yolov8x-pose.pt'))

# puts label on image, avoiding overlaps
def _put_label_nonoverlap(img, text, x, y, used_rects):
    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    font_scale = max(0.9, min(2.2, h / 450.0))
    thickness = max(2, int(font_scale * 2))
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, thickness)
    # Initial placement (top-left of text box). Try to place text fully within bounds.
    bx, by = x, max(0, y - th - 6)

    # Clamp horizontally so text box fits: shift left if overflowing
    if bx + tw + 4 >= w:
        bx = max(0, w - (tw + 5))
    if bx - 4 < 0:
        bx = 4
    # Clamp vertically: keep top of box >=0 and bottom <= h-1
    if by - 4 < 0:
        by = 4
    if by + th + 4 >= h:
        by = max(4, h - (th + 5))

    #  check overlaps
    def overlaps(r1, r2):
        (x1, y1, x2, y2) = r1
        (a1, b1, a2, b2) = r2
        return not (x2 < a1 or a2 < x1 or y2 < b1 or b2 < y1)

    # shift label down until no overlaps
    rect = (max(0, bx - 4), max(0, by - 4), min(w - 1, bx + tw + 4), min(h - 1, by + th + 4))
    safety = 0
    direction = 1  # 1 = down, -1 = up
    while any(overlaps(rect, ur) for ur in used_rects) and safety < 24:
        if direction == 1:
            new_by = by + th + 8
            if new_by + th + 4 >= h:
                direction = -1
                new_by = max(4, h - (th + 5))
        else:
            new_by = max(4, by - (th + 8))
            if new_by <= 4:
                direction = 1
                new_by = min(h - th - 6, by + th + 8)
        by = new_by
        rect = (max(0, bx - 4), max(0, by - 4), min(w - 1, bx + tw + 4), min(h - 1, by + th + 4))
        safety += 1

    # draw label 
    cv2.rectangle(img, (rect[0], rect[1]), (rect[2], rect[3]), (0, 0, 0), -1)
    cv2.putText(img, text, (bx, by + th), font, font_scale, (0, 255, 255), thickness, lineType=cv2.LINE_AA)
    used_rects.append(rect)

def posture_features_from_coords(coords):
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])
    ls, rs = get(5), get(6)
    lh, rh = get(11), get(12)
    nose = get(0)
    shoulder_mid = np.nanmean(np.vstack([ls, rs]), axis=0)
    hip_mid = np.nanmean(np.vstack([lh, rh]), axis=0)
    torso = shoulder_mid - hip_mid
    vertical = np.array([0.0, -1.0])
    
    #get angle of torso and head
    def angle_deg(u, v):
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return np.nan
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return np.nan
        cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return np.degrees(np.arccos(cos))
    
    # compute final features
    torso_tilt = angle_deg(torso, vertical)
    head_vec = nose - shoulder_mid
    head_tilt = angle_deg(head_vec, vertical)

    # returns torso and head tilt angles
    return np.array([torso_tilt, head_tilt])

def gesture_features_from_coords(coords):
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])
    ls, rs = get(5), get(6)
    lw, rw = get(9), get(10)
    nose = get(0)
    left_raise = (lw[1] - ls[1]) if not np.isnan(lw[1]) and not np.isnan(ls[1]) else np.nan
    right_raise = (rw[1] - rs[1]) if not np.isnan(rw[1]) and not np.isnan(rs[1]) else np.nan
    # compute eucleidean distances
    def dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.linalg.norm(a - b))
    left_nose_dist = dist(lw, nose)
    right_nose_dist = dist(rw, nose)
    return np.array([left_raise, right_raise, left_nose_dist, right_nose_dist])

def engagement_features_from_coords(coords):
    # Match training's compute_engagement_features for consistency
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])

    nose = get(0)
    ls, rs = get(5), get(6)
    lh, rh = get(11), get(12)
    lw, rw = get(9), get(10)

    shoulder_mid = np.nanmean(np.vstack([ls, rs]), axis=0)
    hip_mid = np.nanmean(np.vstack([lh, rh]), axis=0)

    def safe_dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.linalg.norm(a - b))

    shoulder_width = safe_dist(ls, rs)
    scale = shoulder_width if shoulder_width and not np.isnan(shoulder_width) and shoulder_width > 0 else np.nan

    head_vec = nose - shoulder_mid
    vertical = np.array([0.0, -1.0])

    #compute angle
    def angle_deg(u, v):
        if np.any(np.isnan(u)) or np.any(np.isnan(v)):
            return np.nan
        nu = np.linalg.norm(u); nv = np.linalg.norm(v)
        if nu == 0 or nv == 0:
            return np.nan
        cos = np.clip(np.dot(u, v) / (nu * nv), -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    head_tilt = angle_deg(head_vec, vertical)

    lnose = safe_dist(lw, nose)
    rnose = safe_dist(rw, nose)
    lhip = safe_dist(lw, hip_mid)
    rhip = safe_dist(rw, hip_mid)

    #normalize distances
    def norm(v):
        return v / scale if scale and not np.isnan(scale) and scale > 0 and not np.isnan(v) else np.nan
    lnose_n = norm(lnose)
    rnose_n = norm(rnose)
    lhip_n = norm(lhip)
    rhip_n = norm(rhip)

    left_wrist_rel = (lw[1] - shoulder_mid[1]) if not np.isnan(lw[1]) and not np.isnan(shoulder_mid[1]) else np.nan
    right_wrist_rel = (rw[1] - shoulder_mid[1]) if not np.isnan(rw[1]) and not np.isnan(shoulder_mid[1]) else np.nan
    left_wrist_rel_n = norm(abs(left_wrist_rel))
    right_wrist_rel_n = norm(abs(right_wrist_rel))

    # Phone proximity features omitted in inference unless later added
    lw_phone_n = np.nan
    rw_phone_n = np.nan

    # Returns: [Head Vector X, Head Vector Y, Head Tilt, Normalized Distances (4), Normalized Wrist Relatives (2), Phone Distances (2)]
    return np.array([
        head_vec[0] if not np.isnan(head_vec[0]) else np.nan,
        head_vec[1] if not np.isnan(head_vec[1]) else np.nan,
        head_tilt,
        lnose_n, rnose_n,
        lhip_n, rhip_n,
        left_wrist_rel_n, right_wrist_rel_n,
        lw_phone_n, rw_phone_n
    ])


def _aux_pose_metrics(coords):
    # Helper metrics for simple rule-based guards (normalized by shoulder width)
    def get(idx):
        return coords[idx] if idx < len(coords) else np.array([np.nan, np.nan])
    nose = get(0)
    leye = get(1)
    reye = get(2)
    lear = get(3)
    rear = get(4)
    ls, rs = get(5), get(6)
    le, re = get(7), get(8)
    lw, rw = get(9), get(10)
    lh, rh = get(11), get(12)
    la, ra = get(15), get(16)

    def safe_dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.linalg.norm(a - b))

    shoulder_width = safe_dist(ls, rs)
    scale = shoulder_width if shoulder_width and not np.isnan(shoulder_width) and shoulder_width > 0 else np.nan

    # Wrist/elbow above shoulder metrics (positive when above shoulders)
    shoulder_mid_y = np.nanmean([ls[1], rs[1]]) if not np.isnan(ls[1]) and not np.isnan(rs[1]) else np.nan
    def norm(v):
        return (v / scale) if scale and not np.isnan(scale) and scale > 0 and not np.isnan(v) else np.nan
    wrist_ups = []
    for w in (lw, rw):
        if not np.isnan(w[1]) and not np.isnan(shoulder_mid_y):
            # y grows downward; wrist above shoulder => negative (w[1] < shoulder_mid_y)
            wrist_ups.append(norm(shoulder_mid_y - w[1]))
    wrist_above_norm = np.nanmax(wrist_ups) if len(wrist_ups) > 0 else np.nan

    elbow_ups = []
    for e in (le, re):
        if not np.isnan(e[1]) and not np.isnan(shoulder_mid_y):
            elbow_ups.append(norm(shoulder_mid_y - e[1]))
    elbow_above_norm = np.nanmax(elbow_ups) if len(elbow_ups) > 0 else np.nan

    # Leg extension metric (ankle sufficiently below hip)
    leg_exts = []
    for hip, ank in ((lh, la), (rh, ra)):
        if not np.isnan(hip[1]) and not np.isnan(ank[1]):
            leg_exts.append(norm(ank[1] - hip[1]))
    leg_extension_norm = float(np.nanmean(leg_exts)) if len(leg_exts) > 0 else np.nan

    # Wrist-to-face (nose/eyes) min distance (normalized)
    def safe_dist(a, b):
        if np.any(np.isnan(a)) or np.any(np.isnan(b)):
            return np.nan
        return float(np.linalg.norm(a - b))
    wf_dists = []
    for w in (lw, rw):
        for fpt in (nose, leye, reye):
            d = safe_dist(w, fpt)
            wf_dists.append(d)
    min_wf = np.nanmin(wf_dists) if len(wf_dists) > 0 else np.nan
    min_wf_n = (min_wf / scale) if scale and not np.isnan(scale) and scale > 0 and not np.isnan(min_wf) else np.nan

    # Wrist-to-ear min distance (normalized) — helps disambiguate cover_ears vs raise
    we_dists = []
    for w in (lw, rw):
        for ept in (lear, rear):
            d = safe_dist(w, ept)
            we_dists.append(d)
    min_we = np.nanmin(we_dists) if len(we_dists) > 0 else np.nan
    min_we_n = (min_we / scale) if scale and not np.isnan(scale) and scale > 0 and not np.isnan(min_we) else np.nan

    return scale, wrist_above_norm, leg_extension_norm, min_wf_n, elbow_above_norm, min_we_n

# process an image and predict cues
def predict_image(image_path, show=True, max_width=1280, max_height=720, save_path=None,
                  guards=False, guard_gesture=0.2, guard_posture=0.7, chin_thresh=0.5, debug_guards=False):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    results = pose_model(img)

    for r in results:
        # draw skeleton/annotated image if available
        try:
            annotated = r.plot()
            out_img = annotated
        except Exception:
            out_img = img.copy()

        if hasattr(r, 'keypoints') and r.keypoints is not None:
            kp = r.keypoints.xy  # tensor (N_people, N_kp, 2 or 3)
            if len(kp) > 0:
                arr = kp.cpu().numpy()
                used_rects = []
                h, w = out_img.shape[:2]
                #for every person detected
                for i, person in enumerate(arr):
                    coords = person[:, :2] if person.ndim == 2 else person
                    
                    # predict posture
                    pf = posture_features_from_coords(coords).reshape(1, -1)
                    pf = np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with open(str(INFER_PATH / 'classifier_posture.pkl'), 'rb') as f:
                            clf_post = pickle.load(f)
                        with open(str(INFER_PATH / 'label_posture.pkl'), 'rb') as f:
                            le_post = pickle.load(f)
                        posture_label = le_post.inverse_transform(clf_post.predict(pf))[0]
                    except Exception:
                        posture_label = 'posture:unknown'
                    
                    # predict gesture
                    gf = gesture_features_from_coords(coords).reshape(1, -1)
                    gf = np.nan_to_num(gf, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with open(str(INFER_PATH / 'classifier_gesture.pkl'), 'rb') as f:
                            clf_gest = pickle.load(f)
                        with open(str(INFER_PATH / 'label_gesture.pkl'), 'rb') as f:
                            le_gest = pickle.load(f)
                        if hasattr(clf_gest, 'predict_proba'):
                            probs = clf_gest.predict_proba(gf)[0]
                            max_p = float(np.max(probs))
                            pred = clf_gest.predict(gf)
                            pred_label = le_gest.inverse_transform(pred)[0]
                            if max_p >= 0.3:
                                gesture_label = pred_label
                            else:
                                # 'neutral' over unknown when available
                                try:
                                    if 'neutral' in set(le_gest.classes_):
                                        gesture_label = 'neutral'
                                    else:
                                        gesture_label = 'gesture:unknown'
                                except Exception:
                                    gesture_label = 'gesture:unknown'
                        else:
                            gesture_label = le_gest.inverse_transform(clf_gest.predict(gf))[0]
                    except Exception:
                        gesture_label = 'gesture:unknown'
                    
                    # predict engagement
                    ef = engagement_features_from_coords(coords).reshape(1, -1)
                    ef = np.nan_to_num(ef, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with open(str(INFER_PATH / 'classifier_engagement.pkl'), 'rb') as f:
                            clf_eng = pickle.load(f)
                        with open(str(INFER_PATH / 'label_engagement.pkl'), 'rb') as f:
                            le_eng = pickle.load(f)
                        if hasattr(clf_eng, 'predict_proba'):
                            probs = clf_eng.predict_proba(ef)[0]
                            max_p = float(np.max(probs))
                            pred = clf_eng.predict(ef)
                            engagement_label = le_eng.inverse_transform(pred)[0] if max_p >= 0.2 else 'engagement:unknown'
                        else:
                            engagement_label = le_eng.inverse_transform(clf_eng.predict(ef))[0]
                    except Exception:
                        engagement_label = 'engagement:unknown'

                    # rule-based guards
                    if guards:
                        _, wrist_above_norm, leg_extension_norm, wrist_face_min, elbow_above_norm, wrist_ear_min = _aux_pose_metrics(coords)
                        # Guard hand_raise: require wrist above shoulder by min normalized amount
                        if isinstance(gesture_label, str) and 'hand' in gesture_label and 'raise' in gesture_label:
                            # Require both wrist and elbow above threshold to confirm a raise
                            if (
                                np.isnan(wrist_above_norm) or np.isnan(elbow_above_norm) or
                                wrist_above_norm < guard_gesture or elbow_above_norm < guard_gesture
                            ):
                                try:
                                    with open(str(INFER_PATH / 'label_gesture.pkl'), 'rb') as f:
                                        _le_tmp = pickle.load(f)
                                    gesture_label = 'neutral' if 'neutral' in set(_le_tmp.classes_) else 'gesture:unknown'
                                except Exception:
                                    gesture_label = 'gesture:unknown'
                        # If model predicts neutral or cover_ears but raise geometry is strong, bias to hand_raise
                        else:
                            try:
                                with open(str(INFER_PATH / 'label_gesture.pkl'), 'rb') as f:
                                    _le_bias = pickle.load(f)
                                classes = set(_le_bias.classes_)
                                strong_raise = (
                                    not np.isnan(wrist_above_norm) and not np.isnan(elbow_above_norm) and
                                    wrist_above_norm >= guard_gesture and elbow_above_norm >= guard_gesture
                                )
                                cover_ears_like = isinstance(gesture_label, str) and ('cover_ears' in gesture_label)
                                # Avoid biasing to raise when wrist is actually very close to ear
                                far_from_ear = (wrist_ear_min is np.nan) or (wrist_ear_min > (guard_gesture * 0.8))
                                if strong_raise and far_from_ear and (
                                    (isinstance(gesture_label, str) and gesture_label == 'neutral') or cover_ears_like
                                ) and ('hand_raise' in classes):
                                    gesture_label = 'hand_raise'
                            except Exception:
                                pass
                        # Hand on chin heuristic: if wrist close to nose, bias to hand_on_chin
                        try:
                            if not np.isnan(wrist_face_min) and wrist_face_min <= chin_thresh:
                                with open(str(INFER_PATH / 'label_gesture.pkl'), 'rb') as f:
                                    _le_tmp2 = pickle.load(f)
                                if 'hand_on_chin' in set(_le_tmp2.classes_):
                                    gesture_label = 'hand_on_chin'
                            if debug_guards:
                                print(f"[guard] image={os.path.basename(image_path)} person={i} wrist_face_min={wrist_face_min:.2f} chin_thresh={chin_thresh} -> {gesture_label}")
                        except Exception:
                            pass
                        # Guard sitting: if legs clearly extended (standing-like), avoid sitting
                        if posture_label == 'sitting':
                            if not np.isnan(leg_extension_norm) and leg_extension_norm >= guard_posture:
                                posture_label = 'standing'

                    label = f"{posture_label} | {gesture_label} | {engagement_label}"

                    # anchor label inside bounding box if available, else near nose
                    x, y = 10, 30
                    try:
                        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > i:
                            b = r.boxes.xyxy[i].cpu().numpy().flatten()
                            # place label just below top-left percentage text using dynamic offset based on text height
                            x = int(max(2, min(w - 2, b[0] + 6)))
                            # estimate text height for current image scale
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = max(0.9, min(2.2, h / 450.0))
                            thickness = max(2, int(font_scale * 2))
                            (_, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            y = int(max(2, min(h - 2, b[1] + th + 10)))
                        elif coords.shape[0] >= 1:
                            x = int(max(2, min(w - 2, coords[0][0])))
                            y = int(max(2, min(h - 2, coords[0][1] - 10)))
                    except Exception:
                        if coords.shape[0] >= 1:
                            x = int(max(2, min(w - 2, coords[0][0])))
                            y = int(max(2, min(h - 2, coords[0][1] - 10)))
                    _put_label_nonoverlap(out_img, label, x, y, used_rects)

        if show:
            h, w = out_img.shape[:2]
            scale = min(max_width / float(w), max_height / float(h), 1.0)
            display = out_img
            if scale < 1.0:
                new_w, new_h = int(w * scale), int(h * scale)
                display = cv2.resize(out_img, (new_w, new_h), interpolation=cv2.INTER_AREA)
            cv2.namedWindow('Non-Verbal Cues', cv2.WINDOW_NORMAL)
            cv2.imshow('Non-Verbal Cues', display)
            cv2.waitKey(0)

        if save_path:
            import os
            out_p = save_path
            # if a directory provided, compose filename
            if os.path.isdir(out_p):
                base = os.path.splitext(os.path.basename(image_path))[0]
                out_p = os.path.join(out_p, base + '_cues.jpg')
            cv2.imwrite(out_p, out_img)

    return results


def run_webcam(max_width=1280, max_height=720, guards=False, guard_gesture=0.2, guard_posture=0.7,
               chin_thresh=0.5, debug_guards=False):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError('Cannot open webcam')

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = pose_model(frame)
        out = frame.copy()
        for r in results:
            try:
                out = r.plot()
            except Exception:
                out = frame

            if hasattr(r, 'keypoints') and r.keypoints is not None:
                kp = r.keypoints.xy
                arr = kp.cpu().numpy()
                used_rects = []
                h, w = out.shape[:2]
                for i, person in enumerate(arr):
                    coords = person[:, :2] if person.ndim == 2 else person
                    # posture
                    pf = posture_features_from_coords(coords).reshape(1, -1)
                    pf = np.nan_to_num(pf, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with open(os.path.join(HERE, 'inference', 'classifier_posture.pkl'), 'rb') as f:
                            clf_post = pickle.load(f)
                        with open(os.path.join(HERE, 'inference', 'label_posture.pkl'), 'rb') as f:
                            le_post = pickle.load(f)
                        posture_label = le_post.inverse_transform(clf_post.predict(pf))[0]
                    except Exception:
                        posture_label = 'posture:unknown'
                    # gesture
                    gf = gesture_features_from_coords(coords).reshape(1, -1)
                    gf = np.nan_to_num(gf, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with open(os.path.join(HERE, 'inference', 'classifier_gesture.pkl'), 'rb') as f:
                            clf_gest = pickle.load(f)
                        with open(os.path.join(HERE, 'inference', 'label_gesture.pkl'), 'rb') as f:
                            le_gest = pickle.load(f)
                        if hasattr(clf_gest, 'predict_proba'):
                            probs = clf_gest.predict_proba(gf)[0]
                            max_p = float(np.max(probs))
                            pred = clf_gest.predict(gf)
                            pred_label = le_gest.inverse_transform(pred)[0]
                            if max_p >= 0.3:
                                gesture_label = pred_label
                            else:
                                try:
                                    gesture_label = 'neutral' if 'neutral' in set(le_gest.classes_) else 'gesture:unknown'
                                except Exception:
                                    gesture_label = 'gesture:unknown'
                        else:
                            gesture_label = le_gest.inverse_transform(clf_gest.predict(gf))[0]
                    except Exception:
                        gesture_label = 'gesture:unknown'
                    # engagement
                    ef = engagement_features_from_coords(coords).reshape(1, -1)
                    ef = np.nan_to_num(ef, nan=0.0, posinf=0.0, neginf=0.0)
                    try:
                        with open(os.path.join(HERE, 'inference', 'classifier_engagement.pkl'), 'rb') as f:
                            clf_eng = pickle.load(f)
                        with open(os.path.join(HERE, 'inference', 'label_engagement.pkl'), 'rb') as f:
                            le_eng = pickle.load(f)
                        if hasattr(clf_eng, 'predict_proba'):
                            probs = clf_eng.predict_proba(ef)[0]
                            max_p = float(np.max(probs))
                            pred = clf_eng.predict(ef)
                            engagement_label = le_eng.inverse_transform(pred)[0] if max_p >= 0.2 else 'engagement:unknown'
                        else:
                            engagement_label = le_eng.inverse_transform(clf_eng.predict(ef))[0]
                    except Exception:
                        engagement_label = 'engagement:unknown'

                    # Optional guards in webcam too
                    if guards:
                        _, wrist_above_norm, leg_extension_norm, wrist_face_min, elbow_above_norm, wrist_ear_min = _aux_pose_metrics(coords)
                        if isinstance(gesture_label, str) and 'hand' in gesture_label and 'raise' in gesture_label:
                            if (
                                np.isnan(wrist_above_norm) or np.isnan(elbow_above_norm) or
                                wrist_above_norm < guard_gesture or elbow_above_norm < guard_gesture
                            ):
                                try:
                                    with open(os.path.join(HERE, 'inference', 'label_gesture.pkl'), 'rb') as f:
                                        _le_tmp = pickle.load(f)
                                    gesture_label = 'neutral' if 'neutral' in set(_le_tmp.classes_) else 'gesture:unknown'
                                except Exception:
                                    gesture_label = 'gesture:unknown'
                        else:
                            try:
                                with open(os.path.join(HERE, 'inference', 'label_gesture.pkl'), 'rb') as f:
                                    _le_bias = pickle.load(f)
                                classes = set(_le_bias.classes_)
                                strong_raise = (
                                    not np.isnan(wrist_above_norm) and not np.isnan(elbow_above_norm) and
                                    wrist_above_norm >= guard_gesture and elbow_above_norm >= guard_gesture
                                )
                                cover_ears_like = isinstance(gesture_label, str) and ('cover_ears' in gesture_label)
                                far_from_ear = (wrist_ear_min is np.nan) or (wrist_ear_min > (guard_gesture * 0.8))
                                if strong_raise and far_from_ear and (
                                    (isinstance(gesture_label, str) and gesture_label == 'neutral') or cover_ears_like
                                ) and ('hand_raise' in classes):
                                    gesture_label = 'hand_raise'
                            except Exception:
                                pass
                        # Hand on chin heuristic
                        try:
                            if not np.isnan(wrist_face_min) and wrist_face_min <= chin_thresh:
                                with open(os.path.join(HERE, 'inference', 'label_gesture.pkl'), 'rb') as f:
                                    _le_tmp2 = pickle.load(f)
                                if 'hand_on_chin' in set(_le_tmp2.classes_):
                                    gesture_label = 'hand_on_chin'
                            if debug_guards:
                                print(f"[guard-webcam] wrist_face_min={wrist_face_min:.2f} chin_thresh={chin_thresh} -> {gesture_label}")
                        except Exception:
                            pass
                        if posture_label == 'sitting':
                            if not np.isnan(leg_extension_norm) and leg_extension_norm >= guard_posture:
                                posture_label = 'standing'

                    label = f"{posture_label} | {gesture_label} | {engagement_label}"
                    # anchor label inside bounding box if available, else near nose
                    x, y = 10, 30
                    try:
                        if hasattr(r, 'boxes') and r.boxes is not None and len(r.boxes) > i:
                            b = r.boxes.xyxy[i].cpu().numpy().flatten()
                            # place label just below top-left percentage text using dynamic offset based on text height
                            x = int(max(2, min(w - 2, b[0] + 6)))
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            font_scale = max(0.9, min(2.2, h / 450.0))
                            thickness = max(2, int(font_scale * 2))
                            (_, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                            y = int(max(2, min(h - 2, b[1] + th + 10)))
                        elif coords.shape[0] >= 1:
                            x = int(max(2, min(w - 2, coords[0][0])))
                            y = int(max(2, min(h - 2, coords[0][1] - 10)))
                    except Exception:
                        if coords.shape[0] >= 1:
                            x = int(max(2, min(w - 2, coords[0][0])))
                            y = int(max(2, min(h - 2, coords[0][1] - 10)))
                    _put_label_nonoverlap(out, label, x, y, used_rects)

        h, w = out.shape[:2]
        scale = min(max_width / float(w), max_height / float(h), 1.0)
        display = out
        if scale < 1.0:
            new_w, new_h = int(w * scale), int(h * scale)
            display = cv2.resize(out, (new_w, new_h), interpolation=cv2.INTER_AREA)
        cv2.namedWindow('Non-Verbal Cues (webcam)', cv2.WINDOW_NORMAL)
        cv2.imshow('Non-Verbal Cues (webcam)', display)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', help='Path to image to run inference on')
    parser.add_argument('--folder', help='Process all images in a folder')
    parser.add_argument('--webcam', action='store_true', help='Run webcam live inference')
    parser.add_argument('--max-width', type=int, default=1280, help='Max display width')
    parser.add_argument('--max-height', type=int, default=720, help='Max display height')
    parser.add_argument('--guards', action='store_true', help='Enable simple rule-based guards for gesture/posture')
    parser.add_argument('--guard-gesture', type=float, default=0.2, help='Min normalized wrist-above-shoulder for hand_raise')
    parser.add_argument('--guard-posture', type=float, default=0.7, help='Min normalized hip-to-ankle for standing vs sitting')
    parser.add_argument('--chin-thresh', type=float, default=0.35, help='Max normalized wrist-to-nose distance to bias hand_on_chin')
    parser.add_argument('--debug-guards', action='store_true', help='Print guard heuristics for tuning')
    parser.add_argument('--save', help='Optional output file or directory to save annotated results')
    parser.add_argument('--no-show', action='store_true', help='Do not show image windows')
    args = parser.parse_args()

    if args.webcam:
        run_webcam(max_width=args.max_width, max_height=args.max_height, guards=args.guards, guard_gesture=args.guard_gesture, guard_posture=args.guard_posture, chin_thresh=args.chin_thresh, debug_guards=args.debug_guards)
    elif args.folder:
        import os
        import glob
        exts = ('*.jpg', '*.jpeg', '*.png')
        files = []
        for e in exts:
            files.extend(glob.glob(os.path.join(args.folder, e)))
        if not files:
            print('No images found in folder:', args.folder)
        else:
            if args.save and os.path.isdir(args.save):
                os.makedirs(args.save, exist_ok=True)
            for fp in sorted(files):
                try:
                    predict_image(fp, show=not args.no_show, max_width=args.max_width, max_height=args.max_height, save_path=args.save, guards=args.guards, guard_gesture=args.guard_gesture, guard_posture=args.guard_posture, chin_thresh=args.chin_thresh, debug_guards=args.debug_guards)
                except Exception as e:
                    print('Error on', fp, e)
    elif args.image:
        predict_image(args.image, show=not args.no_show, max_width=args.max_width, max_height=args.max_height, save_path=args.save, guards=args.guards, guard_gesture=args.guard_gesture, guard_posture=args.guard_posture, chin_thresh=args.chin_thresh, debug_guards=args.debug_guards)
    else:
        print('Provide --image <path>, --folder <dir>, or --webcam')
