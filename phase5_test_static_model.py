"""
Test the trained static ASL classifier on the MS-ASL test images.

Dataset structure:

asl_dataset/
└── test/
    ├── 0/
    ├── 1/
    ├── ...
    ├── 9/
    ├── A/
    ├── B/
    ├── ...
    └── Z/

The test images are converted into the same 63 normalized
hand-landmark features used during training.
"""

from pathlib import Path
import json

import cv2
import joblib
import mediapipe as mp
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

import matplotlib.pyplot as plt

from utils.landmark_utils import extract_features_from_landmarks


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

TEST_DIR = Path("asl_dataset/test")

MODEL_FILE = Path("models/sign_classifier.joblib")
LABELS_FILE = Path("models/labels.json")


# ---------------------------------------------------------
# Settings
# ---------------------------------------------------------

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


# ---------------------------------------------------------
# Load model
# ---------------------------------------------------------

print("Loading model...")

model = joblib.load(MODEL_FILE)
model.n_jobs = 1

print(f"Model loaded: {MODEL_FILE}")


# ---------------------------------------------------------
# Load labels
# ---------------------------------------------------------

with open(LABELS_FILE, "r", encoding="utf-8") as file:
    labels_data = json.load(file)


# phase3_train_classifier.py saves labels as an ordered list, where each
# index is the numeric class ID used by the model.
if isinstance(labels_data, list):
    label_map = {
        index: value
        for index, value in enumerate(labels_data)
    }
elif isinstance(labels_data, dict):
    # Keep compatibility with older dictionary-form labels files.
    label_map = {
        int(key): value
        for key, value in labels_data.items()
    }
else:
    raise ValueError("labels.json must contain a list or object")


# ---------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5,
)


# ---------------------------------------------------------
# Storage for results
# ---------------------------------------------------------

y_true = []
y_pred = []

total_images = 0
successful_images = 0
failed_images = 0


# ---------------------------------------------------------
# Process dataset
# ---------------------------------------------------------

print()
print("Starting test...")
print(f"Test directory: {TEST_DIR}")
print()


# Sort folders so that 0-9, A-Z are processed consistently.
class_folders = sorted(
    [
        folder
        for folder in TEST_DIR.iterdir()
        if folder.is_dir()
    ],
    key=lambda path: path.name,
)


for class_folder in class_folders:

    actual_label = class_folder.name

    image_files = sorted(
        [
            file
            for file in class_folder.iterdir()
            if file.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )

    print(
        f"Testing class {actual_label}: "
        f"{len(image_files)} images"
    )

    for image_path in image_files:

        total_images += 1

        # -------------------------------------------------
        # Read image
        # -------------------------------------------------

        image = cv2.imread(str(image_path))

        if image is None:
            failed_images += 1
            continue

        # -------------------------------------------------
        # Convert BGR → RGB
        # -------------------------------------------------

        rgb_image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        # -------------------------------------------------
        # MediaPipe hand detection
        # -------------------------------------------------

        results = hands.process(rgb_image)

        if not results.multi_hand_landmarks:
            failed_images += 1
            continue

        # -------------------------------------------------
        # Extract first detected hand
        # -------------------------------------------------

        mp_landmarks = results.multi_hand_landmarks[0].landmark

        landmarks = np.array(
            [
                [landmark.x, landmark.y, landmark.z]
                for landmark in mp_landmarks
            ],
            dtype=np.float32,
        )

        # Make sure we have exactly 21 landmarks.
        if landmarks.shape != (21, 3):
            failed_images += 1
            continue

        # -------------------------------------------------
        # Normalize + flatten → 63 features
        # -------------------------------------------------

        feature_vector = extract_features_from_landmarks(
            landmarks
        )

        if feature_vector is None:
            failed_images += 1
            continue

        feature_vector = np.asarray(
            feature_vector,
            dtype=np.float32,
        )

        if feature_vector.shape != (63,):
            failed_images += 1
            continue

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        prediction = model.predict(
            feature_vector.reshape(1, -1)
        )[0]

        predicted_label = label_map.get(
            int(prediction),
            str(prediction),
        )

        # -------------------------------------------------
        # Store results
        # -------------------------------------------------

        y_true.append(actual_label)
        y_pred.append(predicted_label)

        successful_images += 1


# ---------------------------------------------------------
# Close MediaPipe
# ---------------------------------------------------------

hands.close()


# ---------------------------------------------------------
# Check results
# ---------------------------------------------------------

print()
print("=" * 60)
print("TEST COMPLETE")
print("=" * 60)

print(f"Total images:       {total_images}")
print(f"Successfully tested: {successful_images}")
print(f"Failed/skipped:      {failed_images}")


if not y_true:
    print()
    print("No images were successfully processed.")
    print("Check the dataset path and MediaPipe detection.")
    raise SystemExit


# ---------------------------------------------------------
# Accuracy
# ---------------------------------------------------------

accuracy = accuracy_score(y_true, y_pred)

print()
print(f"Accuracy: {accuracy:.4f}")
print(f"Accuracy: {accuracy * 100:.2f}%")


# ---------------------------------------------------------
# Classification report
# ---------------------------------------------------------

print()
print("=" * 60)
print("CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_true,
        y_pred,
        zero_division=0,
    )
)


# ---------------------------------------------------------
# Confusion matrix
# ---------------------------------------------------------

class_labels = sorted(
    set(y_true) | set(y_pred)
)

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=class_labels,
)


print()
print("=" * 60)
print("CONFUSION MATRIX")
print("=" * 60)

print(cm)


# ---------------------------------------------------------
# Plot confusion matrix
# ---------------------------------------------------------

fig, ax = plt.subplots(
    figsize=(16, 16)
)

image = ax.imshow(cm)

ax.set_title(
    "Static ASL Model - Confusion Matrix"
)

ax.set_xlabel(
    "Predicted Label"
)

ax.set_ylabel(
    "Actual Label"
)

ax.set_xticks(
    np.arange(len(class_labels))
)

ax.set_yticks(
    np.arange(len(class_labels))
)

ax.set_xticklabels(
    class_labels,
    rotation=90,
)

ax.set_yticklabels(
    class_labels,
)

# Write values inside the matrix.
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center",
            fontsize=7,
        )

fig.colorbar(
    image,
    ax=ax,
)

plt.tight_layout()

plt.show()