"""Phase 4: real-time static sign prediction with stability filtering."""

import json
from collections import deque
from pathlib import Path

try:
    import cv2
    import joblib
    import mediapipe as mp
    import numpy as np
except ImportError as error:
    raise SystemExit(
        "Missing dependency. Install the project dependencies with: "
        "python -m pip install -r requirements.txt"
    ) from error

try:
    from utils.landmark_utils import (
        extract_features_from_landmarks,
        extract_landmarks_from_results,
        validate_feature_vector,
    )
except ImportError as error:
    raise SystemExit(
        "Could not import landmark utilities. Ensure utils/landmark_utils.py exists."
    ) from error


WINDOW_TITLE = "Sign Language Translator - Phase 4"
CAMERA_INDEX = 0
MODEL_FILE = Path("models/sign_classifier.joblib")
LABELS_FILE = Path("models/labels.json")

MAX_HANDS = 1
MIN_DETECTION_CONFIDENCE = 0.6
MIN_TRACKING_CONFIDENCE = 0.6
MIN_PREDICTION_CONFIDENCE = 0.55
STABILITY_FRAMES = 8


def load_model_and_labels():
    """Load the trained classifier and label names."""
    if not MODEL_FILE.exists():
        raise SystemExit(
            f"Model not found: {MODEL_FILE}. Run phase3_train_classifier.py first."
        )
    if not LABELS_FILE.exists():
        raise SystemExit(
            f"Labels file not found: {LABELS_FILE}. Run phase3_train_classifier.py first."
        )

    classifier = joblib.load(MODEL_FILE)
    labels = json.loads(LABELS_FILE.read_text(encoding="utf-8"))
    if not labels:
        raise SystemExit(f"Labels file is empty: {LABELS_FILE}")
    return classifier, labels


def predict_sign(classifier, labels, features):
    """Return the predicted label and confidence for one feature vector."""
    feature_row = np.asarray(features, dtype=np.float32).reshape(1, -1)
    predicted_id = int(classifier.predict(feature_row)[0])
    if predicted_id < 0 or predicted_id >= len(labels):
        raise ValueError(f"Predicted class id {predicted_id} has no label entry")
    predicted_label = labels[predicted_id]

    confidence = None
    if hasattr(classifier, "predict_proba"):
        probabilities = classifier.predict_proba(feature_row)[0]
        class_ids = [int(class_id) for class_id in classifier.classes_]
        probability_index = class_ids.index(predicted_id)
        confidence = float(probabilities[probability_index])

    return predicted_label, confidence


def get_stable_prediction(history):
    """Return a stable label when recent predictions agree."""
    if len(history) < STABILITY_FRAMES:
        return None

    labels = [label for label, confidence in history]
    confidences = [confidence for label, confidence in history]
    if len(set(labels)) != 1:
        return None
    if any(confidence < MIN_PREDICTION_CONFIDENCE for confidence in confidences):
        return None
    return labels[-1]


def draw_status(frame, prediction, confidence, stable_prediction, hand_detected):
    """Draw prediction state on the OpenCV frame."""
    status_color = (0, 200, 0) if hand_detected else (0, 0, 255)
    confidence_text = "--" if confidence is None else f"{confidence:.2f}"
    prediction_text = prediction if prediction else "--"
    stable_text = stable_prediction if stable_prediction else "--"

    cv2.rectangle(frame, (0, 0), (460, 145), (20, 20, 20), -1)
    cv2.putText(
        frame,
        f"Hand: {'YES' if hand_detected else 'NO'}",
        (12, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        status_color,
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Prediction: {prediction_text}",
        (12, 65),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Confidence: {confidence_text}",
        (12, 100),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"Stable: {stable_text}",
        (12, 135),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        "Q/ESC: quit | R: reset stable label",
        (12, frame.shape[0] - 16),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )


def main():
    """Open the webcam, run predictions, and display stable results."""
    classifier, labels = load_model_and_labels()
    hands_module = mp.solutions.hands
    drawing_utils = mp.solutions.drawing_utils
    drawing_styles = mp.solutions.drawing_styles

    camera = cv2.VideoCapture(CAMERA_INDEX)
    if not camera.isOpened():
        print(
            f"Error: Could not open webcam at camera index {CAMERA_INDEX}. "
            "Check that it is connected and available to this application."
        )
        camera.release()
        return

    prediction_history = deque(maxlen=STABILITY_FRAMES)
    stable_prediction = None

    try:
        with hands_module.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            model_complexity=0,
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            min_tracking_confidence=MIN_TRACKING_CONFIDENCE,
        ) as hands:
            while True:
                frame_read, frame = camera.read()
                if not frame_read:
                    print("Error: Could not read a frame from the webcam.")
                    break

                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)

                hand_detected = bool(results.multi_hand_landmarks)
                prediction = None
                confidence = None

                if hand_detected:
                    for hand_landmarks in results.multi_hand_landmarks:
                        drawing_utils.draw_landmarks(
                            frame,
                            hand_landmarks,
                            hands_module.HAND_CONNECTIONS,
                            drawing_styles.get_default_hand_landmarks_style(),
                            drawing_styles.get_default_hand_connections_style(),
                        )

                    landmark_arrays = extract_landmarks_from_results(results)
                    if landmark_arrays:
                        features = extract_features_from_landmarks(landmark_arrays[0])
                        if features is not None and validate_feature_vector(features):
                            prediction, confidence = predict_sign(
                                classifier,
                                labels,
                                features,
                            )
                            if confidence is None:
                                confidence = 1.0
                            prediction_history.append((prediction, confidence))
                            next_stable = get_stable_prediction(prediction_history)
                            if next_stable is not None:
                                stable_prediction = next_stable
                else:
                    prediction_history.clear()

                draw_status(
                    frame,
                    prediction,
                    confidence,
                    stable_prediction,
                    hand_detected,
                )
                cv2.imshow(WINDOW_TITLE, frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("r"):
                    prediction_history.clear()
                    stable_prediction = None
    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
