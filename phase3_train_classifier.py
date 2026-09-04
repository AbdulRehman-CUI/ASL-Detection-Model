"""Phase 3: Train a classifier from normalized hand-landmark CSV data."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


DATASET_FILE = Path("data/sign_landmarks_from_images.csv")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "sign_classifier.joblib"
LABELS_FILE = MODEL_DIR / "labels.json"
FEATURE_COUNT = 63


def load_dataset():
    """Load the 63 landmark features and string labels from CSV."""
    if not DATASET_FILE.exists():
        raise SystemExit(f"Dataset not found: {DATASET_FILE}")

    features = []
    labels = []
    with DATASET_FILE.open(newline="", encoding="utf-8") as data_file:
        reader = csv.DictReader(data_file)
        feature_names = [f"feature_{index}" for index in range(FEATURE_COUNT)]
        if reader.fieldnames != feature_names + ["label"]:
            raise ValueError("CSV must contain feature_0 through feature_62 and label")
        for row in reader:
            features.append([float(row[name]) for name in feature_names])
            labels.append(row["label"])

    if not features:
        raise ValueError("Dataset contains no samples")
    return np.asarray(features, dtype=np.float32), np.asarray(labels)


def train_classifier():
    """Train, evaluate, and save a Random Forest classifier."""
    features, labels = load_dataset()
    label_names = sorted(set(labels.tolist()))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    numeric_labels = np.asarray([label_to_id[label] for label in labels])

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        numeric_labels,
        test_size=0.2,
        random_state=42,
        stratify=numeric_labels,
    )

    classifier = RandomForestClassifier(
        n_estimators=300,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    print(f"Samples: {len(features)}")
    print(f"Features: {features.shape[1]}")
    print(f"Labels: {len(label_names)} ({', '.join(label_names)})")
    print("\nEvaluation:")
    print(
        classification_report(
            test_labels,
            predictions,
            labels=list(range(len(label_names))),
            target_names=label_names,
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(classifier, MODEL_FILE)
    LABELS_FILE.write_text(json.dumps(label_names, indent=2) + "\n", encoding="utf-8")
    print(f"Model saved to: {MODEL_FILE}")
    print(f"Labels saved to: {LABELS_FILE}")


if __name__ == "__main__":
    train_classifier()
