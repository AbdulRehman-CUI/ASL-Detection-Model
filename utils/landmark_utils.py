"""Reusable utilities for hand landmark extraction and feature preprocessing.

This module provides functions to extract hand landmarks from MediaPipe results,
normalize them, and convert them to feature vectors for ML training and inference.

The normalization strategy makes features invariant to hand position and scale:
- Landmarks are normalized relative to the wrist position (landmark 0).
- Scale is normalized using the distance from wrist to middle finger MCP.

This preprocessing must be applied consistently during:
1. Phase 2 data collection
2. Phase 3 training
3. Phase 4 real-time inference
"""

import numpy as np


# Landmark indices according to MediaPipe Hands
WRIST_INDEX = 0
MIDDLE_MCP_INDEX = 9  # Middle finger metacarpophalangeal joint

EXPECTED_LANDMARK_COUNT = 21
EXPECTED_FEATURE_SIZE = 63  # 21 landmarks * 3 coordinates (x, y, z)


def extract_landmarks_from_results(results):
    """Extract landmarks from MediaPipe Hands results.

    Args:
        results: MediaPipe Hands.process() result object.

    Returns:
        List of landmark arrays (each is Nx3), or empty list if no hands detected.
        Each landmark array has shape (21, 3) representing [x, y, z] for each point.
    """
    landmark_arrays = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Convert landmarks to numpy array [21, 3]
            landmarks = np.array(
                [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark]
            )
            landmark_arrays.append(landmarks)
    return landmark_arrays


def normalize_landmarks(landmarks):
    """Normalize landmarks relative to wrist position and scale.

    Normalization process:
    1. Translate so wrist (landmark 0) is at origin.
    2. Scale by the distance from wrist to middle finger MCP.

    This makes features invariant to hand position and distance from camera.

    Args:
        landmarks: numpy array of shape (21, 3) representing hand landmarks.

    Returns:
        Normalized landmarks as numpy array of shape (21, 3).

    Raises:
        ValueError: If landmarks don't have the expected shape.
    """
    if landmarks.shape != (EXPECTED_LANDMARK_COUNT, 3):
        raise ValueError(
            f"Expected landmarks shape (21, 3), got {landmarks.shape}"
        )

    # Make a copy to avoid modifying the original
    normalized = landmarks.copy()

    # Step 1: Translate to wrist origin
    wrist = normalized[WRIST_INDEX]
    normalized = normalized - wrist

    # Step 2: Scale normalization
    # Use distance from wrist to middle finger MCP as the reference distance
    reference_distance = np.linalg.norm(
        normalized[MIDDLE_MCP_INDEX] - normalized[WRIST_INDEX]
    )

    # Avoid division by zero
    if reference_distance > 1e-6:
        normalized = normalized / reference_distance

    return normalized


def landmarks_to_features(landmarks):
    """Convert landmarks to a flat feature vector.

    The feature vector is: [x1, y1, z1, x2, y2, z2, ..., x21, y21, z21]

    Args:
        landmarks: numpy array of shape (21, 3).

    Returns:
        Flattened feature vector as numpy array of shape (63,).

    Raises:
        ValueError: If landmarks don't have the expected shape.
    """
    if landmarks.shape != (EXPECTED_LANDMARK_COUNT, 3):
        raise ValueError(
            f"Expected landmarks shape (21, 3), got {landmarks.shape}"
        )
    return landmarks.flatten()


def extract_features_from_landmarks(landmarks):
    """Complete pipeline: normalize landmarks and convert to feature vector.

    Args:
        landmarks: numpy array of shape (21, 3).

    Returns:
        Feature vector as numpy array of shape (63,), or None if normalization fails.
    """
    try:
        normalized = normalize_landmarks(landmarks)
        features = landmarks_to_features(normalized)
        return features
    except ValueError:
        return None


def validate_feature_vector(features):
    """Check if a feature vector has the expected size.

    Args:
        features: numpy array or list of features.

    Returns:
        True if the feature vector has the correct size, False otherwise.
    """
    try:
        features = np.asarray(features)
        return features.size == EXPECTED_FEATURE_SIZE
    except (ValueError, TypeError):
        return False


def get_feature_size():
    """Return the expected size of a feature vector.

    Returns:
        Integer: 63 (21 landmarks * 3 coordinates).
    """
    return EXPECTED_FEATURE_SIZE
