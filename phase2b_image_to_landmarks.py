"""Phase 2B: Convert image datasets to normalized hand-landmark features."""

try:
    import csv
    from pathlib import Path
    import cv2
    import mediapipe as mp
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Install with: "
        "python -m pip install -r requirements.txt"
    ) from error

# Import reusable landmark utilities
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
DATA_DIR = Path("data")
DATASET_FILE = DATA_DIR / "sign_landmarks_from_images.csv"
ASL_DATASET_DIR = Path("asl_dataset")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VALID_IMAGE_LABELS = {str(index) for index in range(10)} | {
    chr(code) for code in range(ord("A"), ord("Z") + 1)
}

# CSV header: 63 feature columns + 1 label column
CSV_HEADER = [f"feature_{i}" for i in range(63)] + ["label"]


def ensure_data_directory():
    """Create data directory if it doesn't exist."""
    DATA_DIR.mkdir(exist_ok=True)


def ensure_csv_header():
    """Start a fresh CSV so repeated runs cannot duplicate samples."""
    try:
        with open(DATASET_FILE, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(CSV_HEADER)
        print(f"✓ Created dataset file: {DATASET_FILE}")
    except IOError as error:
        print(f"✗ Error creating dataset file: {error}")
        return False
    return True


def iter_asl_images():
    """Yield ASL image paths and labels from class-named directories."""
    if not ASL_DATASET_DIR.is_dir():
        return

    for class_dir in sorted(ASL_DATASET_DIR.iterdir(), key=lambda path: path.name):
        if not class_dir.is_dir():
            continue
        label = class_dir.name.upper()
        if label not in VALID_IMAGE_LABELS:
            print(f"Skipping unsupported class folder: {class_dir.name}")
            continue
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                yield image_path, label


def write_sample(writer, features, label):
    """Write one validated feature vector to the open CSV writer."""
    if validate_feature_vector(features):
        writer.writerow(list(features) + [label])
        return True
    return False


def process_images_to_landmarks():
    """Main processing loop: convert images to landmarks and save to CSV."""
    ensure_data_directory()
    if not ensure_csv_header():
        return

    asl_images = list(iter_asl_images())

    print(f"\n{'='*70}")
    print("CONVERTING IMAGES TO LANDMARKS")
    print(f"{'='*70}")
    print(f"ASL folder images: {len(asl_images)}")
    print(f"Expected landmarks per image: 21 (x, y, z)")
    print(f"Expected features per image: 63 (normalized)")
    print(f"{'='*70}\n")

    hands_module = mp.solutions.hands
    
    samples_saved = 0
    samples_failed = 0
    label_counts = {}

    try:
        with hands_module.Hands(
            static_image_mode=True,  # Process as static image, not video
            max_num_hands=1,  # Expect exactly one hand
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        ) as hands:
            with open(DATASET_FILE, mode="a", newline="", encoding="utf-8") as output_file:
                writer = csv.writer(output_file)

                for idx, (image_path, label) in enumerate(asl_images, start=1):
                    if idx % 100 == 0:
                        print(f"Processing ASL: {idx}/{len(asl_images)} images...")

                    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                    if image is None:
                        samples_failed += 1
                        continue

                    results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                    landmark_arrays = extract_landmarks_from_results(results)
                    if not landmark_arrays:
                        samples_failed += 1
                        continue

                    features = extract_features_from_landmarks(landmark_arrays[0])
                    if features is not None and write_sample(writer, features, label):
                        samples_saved += 1
                        label_counts[label] = label_counts.get(label, 0) + 1
                    else:
                        samples_failed += 1

    finally:
        pass

    # Print summary
    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"✓ Samples successfully saved: {samples_saved}")
    print(f"✗ Samples failed (no hand detected): {samples_failed}")
    print(f"  Total processed: {samples_saved + samples_failed}")
    print(f"\nSamples per label:")
    for label_name in sorted(label_counts):
        count = label_counts[label_name]
        bar = "█" * (count // 10)
        print(f"  {label_name:6s}: {count:4d} {bar}")
    print(f"\nDataset saved to: {DATASET_FILE}")
    print(f"{'='*70}\n")

    if samples_saved == 0:
        print("⚠ WARNING: No samples were successfully saved.")
        print("This may indicate:")
        print("  - Images are too small/low quality for MediaPipe")
        print("  - Images don't contain clear hand gestures")
        print("  - MediaPipe detection confidence threshold too high")
        return False

    return True


if __name__ == "__main__":
    try:
        success = process_images_to_landmarks()
        if not success:
            raise SystemExit("Processing failed. Check warnings above.")
    except KeyboardInterrupt:
        print("\n\n⚠ Processing interrupted by user.")
        raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as error:
        print(f"\n✗ Unexpected error: {error}")
        raise SystemExit(1)
