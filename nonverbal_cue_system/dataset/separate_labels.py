import argparse
from pathlib import Path
import re


def parse_line(line: str):
    s = line.strip()
    if not s or s.startswith('#'):
        return None
    # Split by first two commas only (labels may contain commas)
    first = s.find(',')
    if first == -1:
        return None
    second = s.find(',', first + 1)
    if second == -1:
        return None
    fname = s[:first].strip()
    person_idx = s[first + 1:second].strip()
    label = s[second + 1:].strip()
    # Some lines might accidentally include extra commas; keep the tail intact
    return fname, person_idx, label


def norm(text: str) -> str:
    t = text.lower()
    t = t.replace('\t', ' ')
    t = re.sub(r'\s+', ' ', t)
    return t.strip()


def map_posture(label_text: str) -> str:
    t = norm(label_text)
    if not t:
        return ''
    # Priority mapping
    if 'sleep' in t:
        return 'sleeping'
    if 'standing' in t:
        return 'standing'
    if 'slouch' in t:
        return 'slouching'
    if 'arms crossed' in t:
        return 'arms_crossed'
    if 'neutral' in t:
        return 'neutral'
    # fallback
    return ''


def map_gesture(label_text: str) -> str:
    t = norm(label_text)
    if not t:
        return ''
    if 'raising hand' in t or 'raise hand' in t or 'hand raise' in t:
        return 'hand_raise'
    if 'covering ears' in t or 'cover ears' in t:
        return 'cover_ears'
    if 'hand on chin' in t or 'hands on chin' in t:
        return 'hand_on_chin'
    if 'middle finger' in t:
        return 'middle_finger'
    if 'pointing' in t:
        return 'pointing'
    if 'waving' in t:
        return 'waving'
    if 'smiling' in t or 'smile' in t:
        return 'smile'
    if 'writing' in t:
        return 'writing'
    return ''


def map_engagement(label_text: str) -> str:
    t = norm(label_text)
    if not t:
        return ''
    if 'sleep' in t:
        return 'sleeping'
    if 'phone' in t:
        return 'device_phone'
    if 'laptop' in t:
        return 'device_laptop'
    if 'reading' in t:
        return 'reading'
    if 'writing' in t:
        return 'writing'
    if 'talking' in t and 'teacher' not in t:
        return 'talking_seatmate'
    if 'looking at teacher' in t or 'look at teacher' in t:
        return 'looking_teacher'
    if 'looking away' in t or 'looking back' in t or 'look away' in t:
        return 'looking_away'
    return ''


def separate_labels(input_csv: Path, out_posture: Path, out_gesture: Path, out_engagement: Path):
    lines = input_csv.read_text(encoding='utf-8', errors='ignore').splitlines()

    out_posture.parent.mkdir(parents=True, exist_ok=True)
    out_gesture.parent.mkdir(parents=True, exist_ok=True)
    out_engagement.parent.mkdir(parents=True, exist_ok=True)

    p_rows = ['filename,person_index,label']
    g_rows = ['filename,person_index,label']
    e_rows = ['filename,person_index,label']

    for line in lines:
        parsed = parse_line(line)
        if not parsed:
            continue
        fname, person_idx, label_text = parsed
        # Skip obviously broken lines
        if not fname or not person_idx:
            continue
        # Some files include stray commentary like "but"; no-op for mapping
        p = map_posture(label_text)
        g = map_gesture(label_text)
        e = map_engagement(label_text)

        p_rows.append(f"{fname},{person_idx},{p}")
        g_rows.append(f"{fname},{person_idx},{g}")
        e_rows.append(f"{fname},{person_idx},{e}")

    out_posture.write_text('\n'.join(p_rows), encoding='utf-8')
    out_gesture.write_text('\n'.join(g_rows), encoding='utf-8')
    out_engagement.write_text('\n'.join(e_rows), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description='Separate mixed labels into posture, gesture, and engagement CSVs.')
    parser.add_argument('--input', '-i', type=str, default='dataset/labels_posture.csv', help='Path to input CSV with mixed free-text labels')
    parser.add_argument('--outdir', '-o', type=str, default='dataset', help='Output directory for separated CSVs')
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    in_path = (root / args.input).resolve()
    outdir = (root / args.outdir).resolve()

    out_posture = outdir / 'labels_posture_only.csv'
    out_gesture = outdir / 'labels_gesture.csv'
    out_engagement = outdir / 'labels_engagement.csv'

    if not in_path.exists():
        raise FileNotFoundError(f"Input CSV not found: {in_path}")

    separate_labels(in_path, out_posture, out_gesture, out_engagement)
    print(f"Wrote: {out_posture}")
    print(f"Wrote: {out_gesture}")
    print(f"Wrote: {out_engagement}")


if __name__ == '__main__':
    main()
