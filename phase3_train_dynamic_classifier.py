"""Train a classifier for dynamic hand-sign sequences."""

import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


DATASET_DIR = Path("data/dynamic_sequences")
MODEL_DIR = Path("models")
MODEL_FILE = MODEL_DIR / "dynamic_sign_classifier.joblib"
LABELS_FILE = MODEL_DIR / "dynamic_labels.json"
EXPECTED_FEATURES = 63


def load_dataset(dataset_dir):
    """Load and flatten every valid sequence below the label directories."""
    samples = []
    labels = []
    skipped = 0
    expected_frames = None

    for sequence_file in sorted(dataset_dir.glob("*/*.npz")):
        try:
            with np.load(sequence_file, allow_pickle=False) as data:
                if "features" not in data:
                    raise ValueError("missing 'features' array")
                sequence = np.asarray(data["features"], dtype=np.float32)

            if sequence.ndim != 2 or sequence.shape[1] != EXPECTED_FEATURES:
                raise ValueError(
                    f"expected (frames, {EXPECTED_FEATURES}), got {sequence.shape}"
                )
            if sequence.shape[0] < 2:
                raise ValueError("sequence must contain at least 2 frames")
            if expected_frames is None:
                expected_frames = sequence.shape[0]
            elif sequence.shape[0] != expected_frames:
                raise ValueError(
                    f"expected {expected_frames} frames, got {sequence.shape[0]}"
                )
            if not np.isfinite(sequence).all():
                raise ValueError("sequence contains NaN or infinite values")

            # Add frame-to-frame motion so the classifier can distinguish
            # otherwise similar poses that move in different ways.
            motion = np.diff(sequence, axis=0, prepend=sequence[:1])
            samples.append(np.concatenate((sequence, motion), axis=1).ravel())
            labels.append(sequence_file.parent.name)
        except (OSError, ValueError, KeyError) as error:
            skipped += 1
            print(f"Skipping {sequence_file}: {error}")

    if not samples:
        raise SystemExit(f"No valid .npz sequences found in {dataset_dir}")

    return (
        np.asarray(samples, dtype=np.float32),
        np.asarray(labels),
        skipped,
    )


def train_dynamic_classifier(args):
    """Train, evaluate, and save the dynamic sequence classifier."""
    features, labels, skipped = load_dataset(args.dataset_dir)
    label_counts = Counter(labels.tolist())
    skipped_labels = sorted(
        label for label, count in label_counts.items() if count < 2
    )

    if skipped_labels:
        keep_mask = np.asarray(
            [label not in skipped_labels for label in labels],
            dtype=bool,
        )
        features = features[keep_mask]
        labels = labels[keep_mask]
        print(
            "Skipping labels with fewer than 2 sequences: "
            + ", ".join(skipped_labels)
        )

    if not len(labels):
        raise SystemExit("No labels have at least 2 valid sequences")

    label_names = sorted(set(labels.tolist()))
    label_to_id = {label: index for index, label in enumerate(label_names)}
    numeric_labels = np.asarray([label_to_id[label] for label in labels])

    train_features, test_features, train_labels, test_labels = train_test_split(
        features,
        numeric_labels,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=numeric_labels,
    )

    classifier = ExtraTreesClassifier(
        n_estimators=args.estimators,
        random_state=args.random_state,
        n_jobs=-1,
        class_weight="balanced",
    )
    classifier.fit(train_features, train_labels)
    predictions = classifier.predict(test_features)

    print(f"Samples: {len(features)} ({skipped} skipped)")
    print(f"Labels: {len(label_names)}")
    print(f"Input shape per sample: {features.shape[1]} features")
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

    args.model_file.parent.mkdir(parents=True, exist_ok=True)
    args.labels_file.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(classifier, args.model_file)
    args.labels_file.write_text(
        json.dumps(
            {
                "labels": label_names,
                "sequence_features": EXPECTED_FEATURES,
                "uses_motion_features": True,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Model saved to: {args.model_file}")
    print(f"Labels saved to: {args.labels_file}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a separate classifier from dynamic .npz sequences."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DATASET_DIR,
        help=f"Root directory containing label folders (default: {DATASET_DIR})",
    )
    parser.add_argument(
        "--model-file",
        type=Path,
        default=MODEL_FILE,
        help=f"Output model path (default: {MODEL_FILE})",
    )
    parser.add_argument(
        "--labels-file",
        type=Path,
        default=LABELS_FILE,
        help=f"Output label metadata path (default: {LABELS_FILE})",
    )
    parser.add_argument(
        "--estimators",
        type=int,
        default=300,
        help="Number of trees (default: 300)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction reserved for evaluation (default: 0.2)",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train_dynamic_classifier(parse_args())