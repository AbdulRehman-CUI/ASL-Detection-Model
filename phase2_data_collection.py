"""Phase 2: Collect landmark data for sign language classification.

This script allows users to:
1. Select a sign/label to collect data for.
2. Perform the sign in front of the webcam.
3. Let MediaPipe detect and extract hand landmarks.
4. Save landmark features with labels to a CSV dataset.

The collected dataset will be used in Phase 3 for classifier training.

Controls:
- SPACE: Start/pause data collection
- C: Change current sign
- R: Reset sample counter for current sign (without deleting saved data)
- Q / ESC: Quit

Collection strategy:
- Capture approximately every 0.33 seconds (10 frames at ~30 FPS) to avoid duplicates.
- Only save when a hand is successfully detected.
- Use normalized landmarks (relative to wrist position and scale).
- Target 300 samples per sign (configurable).
"""

try:
    import csv
    import time
    from pathlib import Path
    import cv2
    import numpy as np
    import mediapipe as mp
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Install the project dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from error

# Import the reusable landmark utilities
try:
    from utils.landmark_utils import (
        extract_landmarks_from_results,
        extract_features_from_landmarks,
        validate_feature_vector,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import landmark utilities. "
        "Ensure utils/landmark_utils.py exists."
    ) from error


# Configuration
WINDOW_TITLE = "Sign Language Translator - Phase 2: Data Collection"
CAMERA_INDEX = 0
DATA_DIR = Path("data")
DATASET_FILE = DATA_DIR / "sign_landmarks.csv"

# Sampling strategy: capture every N frames (at ~30 FPS, 10 = ~3.3 FPS)
CAPTURE_INTERVAL_FRAMES = 10

# Target samples per sign
TARGET_SAMPLES = 300

# List of signs to collect data for
SIGNS = [
    "HELLO",
    "THANK_YOU",
    "YES",
    "NO",
    "HELP",
    "WATER",
    "STOP",
    "PLEASE",
    "I_LOVE_YOU",
]

# CSV header: 63 feature columns + 1 label column
CSV_HEADER = (
    [f"feature_{i}" for i in range(63)] + ["label"]
)


def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)


def ensure_csv_header():
    """Create CSV file with header if it doesn't exist."""
    if not DATASET_FILE.exists():
        try:
            with open(DATASET_FILE, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(CSV_HEADER)
            print(f"✓ Created dataset file: {DATASET_FILE}")
        except IOError as error:
            print(f"✗ Error creating dataset file: {error}")
            return False
    return True


def select_sign():
    """Display menu and let user select a sign to collect."""
    while True:
        print("\n" + "=" * 60)
        print("SELECT A SIGN TO COLLECT DATA FOR")
        print("=" * 60)
        for i, sign in enumerate(SIGNS, 1):
            print(f"  {i}. {sign}")
        print(f"  0. Exit")
        print("=" * 60)

        try:
            choice = int(input("Enter your choice (0-9): "))
            if choice == 0:
                return None
            if 1 <= choice <= len(SIGNS):
                return SIGNS[choice - 1]
            print("Invalid choice. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def get_existing_sample_count(sign):
    """Count how many samples already exist for a given sign in the dataset."""
    if not DATASET_FILE.exists():
        return 0

    count = 0
    try:
        with open(DATASET_FILE, mode="r", encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            for row in reader:
                if row and row[-1] == sign:
                    count += 1
    except IOError as error:
        print(f"Warning: Could not read dataset file: {error}")
    return count


def save_sample(features, label):
    """Save a sample (features + label) to the CSV file."""
    try:
        with open(DATASET_FILE, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            row = list(features) + [label]
            writer.writerow(row)
        return True
    except IOError as error:
        print(f"Error saving sample: {error}")
        return False


def run_data_collection():
    """Main data collection loop."""
    ensure_data_directory()
    if not ensure_csv_header():
        return

    selected_sign = select_sign()
    if selected_sign is None:
        print("\nExiting data collection.")
        return

    hands_module = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print(
            f"✗ Error: Could not open webcam at camera index {CAMERA_INDEX}. "
            "Check that it is connected and available."
        )
        camera.release()
        return

    print(f"\n✓ Webcam opened successfully.")
    print(f"✓ Collecting data for sign: {selected_sign}")
    print(f"  Press SPACE to start/pause collection")
    print(f"  Press C to change sign")
    print(f"  Press R to reset counter")
    print(f"  Press Q or ESC to quit\n")

    current_sign = selected_sign
    collecting = False
    frame_count = 0
    sample_count = get_existing_sample_count(current_sign)
    last_save_time = time.time()

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=0,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        ) as hands:
            while True:
                frame_read, frame = camera.read()

                if not frame_read:
                    print("✗ Error: Could not read frame from webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                hand_detected = False
                sample_saved = False

                # Process detected hands
                if results.multi_hand_landmarks:
                    hand_detected = True

                    # Use only the first detected hand
                    hand_landmarks = results.multi_hand_landmarks[0]

                    # Draw landmarks
                    drawing_utils.draw_landmarks(
                        frame,
                        hand_landmarks,
                        hands_module.HAND_CONNECTIONS,
                        drawing_styles.get_default_hand_landmarks_style(),
                        drawing_styles.get_default_hand_connections_style(),
                    )

                    # Attempt to extract and save a sample
                    if collecting:
                        landmark_arrays = extract_landmarks_from_results(results)
                        if landmark_arrays:
                            landmarks = landmark_arrays[0]
                            features = extract_features_from_landmarks(landmarks)

                            if features is not None and validate_feature_vector(
                                features
                            ):
                                # Implement sampling strategy: save every N frames
                                current_time = time.time()
                                if (
                                    frame_count % CAPTURE_INTERVAL_FRAMES == 0
                                    or (current_time - last_save_time) >= 0.3
                                ):
                                    if save_sample(features, current_sign):
                                        sample_count += 1
                                        sample_saved = True
                                        last_save_time = current_time

                frame_count += 1

                # Prepare status text
                status_color = (0, 255, 0)  # Green
                if not hand_detected:
                    status_color = (0, 0, 255)  # Red

                # Draw UI text on frame
                cv2.putText(
                    frame,
                    "SIGN LANGUAGE DATA COLLECTION - PHASE 2",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Sign: {current_sign}",
                    (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )
                cv2.putText(
                    frame,
                    f"Samples: {sample_count} / {TARGET_SAMPLES}",
                    (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

                # Collection status
                collection_text = "COLLECTING" if collecting else "PAUSED"
                collection_color = (0, 255, 0) if collecting else (255, 165, 0)
                cv2.putText(
                    frame,
                    f"Status: {collection_text}",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    collection_color,
                    2,
                )

                # Hand detection status
                hand_text = "Hand: YES" if hand_detected else "Hand: NO"
                cv2.putText(
                    frame,
                    hand_text,
                    (10, 190),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    status_color,
                    2,
                )

                # Show if sample was saved
                if sample_saved:
                    cv2.putText(
                        frame,
                        "SAMPLE SAVED",
                        (10, 230),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Target reached indicator
                if sample_count >= TARGET_SAMPLES:
                    cv2.putText(
                        frame,
                        f"TARGET REACHED FOR {current_sign}!",
                        (10, 270),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )

                # Controls
                cv2.putText(
                    frame,
                    "CONTROLS: SPACE=Start/Pause  C=Change  R=Reset  Q=Quit",
                    (10, frame.shape[0] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                cv2.imshow(WINDOW_TITLE, frame)

                # Handle user input
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):  # Q or ESC
                    print(
                        f"\n✓ Exiting. Saved {sample_count} samples for {current_sign}."
                    )
                    break

                if key == ord(" "):  # Space
                    collecting = not collecting
                    action = "Started" if collecting else "Paused"
                    print(f"{action} collection for {current_sign}.")

                if key == ord("c"):  # C
                    new_sign = select_sign()
                    if new_sign is not None:
                        current_sign = new_sign
                        sample_count = get_existing_sample_count(current_sign)
                        collecting = False
                        frame_count = 0
                        print(f"\n✓ Switched to sign: {current_sign}")
                        print(f"  Existing samples: {sample_count}")
                    else:
                        break

                if key == ord("r"):  # R
                    sample_count = 0
                    print(
                        f"✓ Reset sample counter for {current_sign} (data not deleted)."
                    )

    finally:
        camera.release()
        cv2.destroyAllWindows()

    print(f"\n✓ Data collection complete!")
    print(f"✓ Dataset saved to: {DATASET_FILE}")
    print(f"✓ Ready for Phase 3 (classifier training).\n")


if __name__ == "__main__":
    run_data_collection()
