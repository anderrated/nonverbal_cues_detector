# Trains the classification models (Random Forest) for posture, gesture, and engagement
'''
python nonverbal_cue_system/training/train_classifier.py --posture nonverbal_cue_system/training/posture_features.csv --gesture nonverbal_cue_system/training/gesture_features.csv --engagement nonverbal_cue_system/training/engagement_features.csv --outdir nonverbal_cue_system/inference
'''

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from collections import Counter
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# separate features and labels from CSV
def load_dataset(csv_path):
    df = pd.read_csv(csv_path, header=None)
    # Last column is label
    X = df.iloc[:, :-1].values
    y = df.iloc[:, -1].values
    return X, y


def train_and_save(csv_path, out_model_path, out_label_path):
    X, y = load_dataset(csv_path)

    # Encode labels
    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    # Only stratify if all classes have >=2 samples
    counts = Counter(y_enc)
    can_stratify = all(c >= 2 for c in counts.values()) and len(X) >= 5
    stratify_arg = y_enc if can_stratify else None
    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=stratify_arg)
    
    # Train Random Forest Classifier Model
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_train, y_train)

    # Evaluation
    y_pred = clf.predict(X_test)
    print(f"Report for {csv_path}:")
    try:
        print(classification_report(y_test, y_pred, target_names=le.classes_))
    except Exception:
        print(classification_report(y_test, y_pred))
    
    # Save the trained classifier and label encoder
    with open(out_model_path, 'wb') as f:
        pickle.dump(clf, f)
    with open(out_label_path, 'wb') as f:
        pickle.dump(le, f)
    print(f"Saved model to {out_model_path} and labels to {out_label_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train classifiers for posture, gesture, and engagement.')
    parser.add_argument('--posture', help='CSV for posture features')
    parser.add_argument('--gesture', help='CSV for gesture features')
    parser.add_argument('--engagement', help='CSV for engagement features')
    parser.add_argument('--outdir', default=os.path.join(os.path.dirname(__file__), '..', 'inference'), help='Where to save models')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    if args.posture:
        train_and_save(args.posture, os.path.join(args.outdir, 'classifier_posture.pkl'), os.path.join(args.outdir, 'label_posture.pkl'))
    if args.gesture:
        train_and_save(args.gesture, os.path.join(args.outdir, 'classifier_gesture.pkl'), os.path.join(args.outdir, 'label_gesture.pkl'))
    if args.engagement:
        train_and_save(args.engagement, os.path.join(args.outdir, 'classifier_engagement.pkl'), os.path.join(args.outdir, 'label_engagement.pkl'))
    if not any([args.posture, args.gesture, args.engagement]):
        print('Provide at least one of --posture, --gesture, --engagement CSV paths.')
