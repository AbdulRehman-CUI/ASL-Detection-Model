"""
Create a small dynamic ASL dataset from MS-ASL using 50 basic words.

Reads MS-ASL annotations, keeps only exact matches from BASIC_WORDS,
downloads only those videos, extracts normalized MediaPipe landmarks,
resamples each sign to 30 frames, and saves .npz sequences.

Examples:
    python dynamic_data_collection_msasl_basic50.py --split train --limit 50
    python dynamic_data_collection_msasl_basic50.py --split train --limit 200
    python dynamic_data_collection_msasl_basic50.py --split val --limit 50

Use --limit 0 for all remaining matching samples.
"""

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
import yt_dlp

from utils.landmark_utils import (
    extract_features_from_landmarks,
    extract_landmarks_from_results,
)


MS_ASL_DIR = Path("MS-ASL")
DATA_DIR = Path("data")
VIDEO_DIR = DATA_DIR / "ms_asl_videos"
SEQUENCE_DIR = DATA_DIR / "dynamic_sequences"
STATE_FILE = DATA_DIR / "ms_asl_basic50_state.json"

TARGET_FRAMES = 30

ANNOTATION_FILES = {
    "train": MS_ASL_DIR / "MSASL_train.json",
    "val": MS_ASL_DIR / "MSASL_val.json",
    "test": MS_ASL_DIR / "MSASL_test.json",
}

# 50 basic/common words for the first dynamic model.
BASIC_WORDS = [
    "hello", "goodbye", "yes", "no", "please", "thank you", "sorry",
    "help", "welcome", "love", "like", "want", "need", "know",
    "understand", "learn", "see", "look", "hear", "speak", "go",
    "come", "stop", "start", "wait", "eat", "drink", "water", "food",
    "home", "school", "work", "friend", "family", "mother", "father",
    "child", "man", "woman", "name", "where", "what", "who", "when",
    "why", "how", "today", "tomorrow", "good", "bad",
]


def normalize_label(text):
    return "_".join(str(text).strip().upper().split())


def normalize_word(text):
    return " ".join(str(text).strip().lower().replace("_", " ").split())


def load_annotations(split):
    path = ANNOTATION_FILES[split]
    if not path.exists():
        raise FileNotFoundError(f"MS-ASL annotation file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_state():
    if not STATE_FILE.exists():
        return {
            "train": {"processed": [], "failed": {}},
            "val": {"processed": [], "failed": {}},
            "test": {"processed": [], "failed": {}},
        }

    with STATE_FILE.open("r", encoding="utf-8") as f:
        state = json.load(f)

    for split in ("train", "val", "test"):
        state.setdefault(split, {"processed": [], "failed": {}})
        state[split].setdefault("processed", [])
        state[split].setdefault("failed", {})

    return state


def save_state(state):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def video_filename(url):
    video_id = hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]
    return VIDEO_DIR / f"{video_id}.mp4"


def download_video(url):
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    output_path = video_filename(url)

    if output_path.exists() and output_path.stat().st_size > 0:
        return output_path

    template = str(output_path.with_suffix(".%(ext)s"))
    
    ydl_opts = {
        "format": "mp4/best",
        "outtmpl": template,
        "quiet": True,
        "no_warnings": True,
        "noplaylist": True,
        "merge_output_format": "mp4",
        }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

    if output_path.exists():
        return output_path

    candidates = list(VIDEO_DIR.glob(output_path.stem + ".*"))
    if not candidates:
        raise FileNotFoundError(f"Downloaded video not found for URL: {url}")
    return candidates[0]


def extract_sequence(video_path, start_time, end_time):
    
    total_frames = 0
    detected_frames = 0
    
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0:
        fps = 30.0

    start_frame = max(0, int(start_time * fps))
    end_frame = max(start_frame + 1, int(end_time * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    features = []

    with mp.solutions.hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:

        frame_number = start_frame

        while frame_number <= end_frame:
            success, frame = cap.read()
            if not success:
                break

            total_frames += 1
            
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            if results.multi_hand_landmarks:
                detected_frames += 1
                
                landmarks = results.multi_hand_landmarks[0].landmark
                
                landmarks = np.array(
                    [[lm.x, lm.y, lm.z] for lm in landmarks],
                    dtype=np.float32,
                )
                 
                if landmarks.shape == (21, 3):
                    
                    feature_vector = extract_features_from_landmarks(landmarks)
                    
                    if feature_vector is not None:
                        features.append(
                            np.asarray(feature_vector, dtype=np.float32)
                        )

            frame_number += 1

    cap.release()

    if len(features) < 2:
        raise RuntimeError(
            f"Not enough landmarks found. "
            f"Video frames read: {total_frames}, "
            f"MediaPipe detections: {detected_frames}, "
            f"valid feature frames: {len(features)}"
        )
    features = np.asarray(features, dtype=np.float32)

    old_positions = np.linspace(0.0, 1.0, len(features))
    new_positions = np.linspace(0.0, 1.0, TARGET_FRAMES)

    resampled = np.empty(
        (TARGET_FRAMES, features.shape[1]), dtype=np.float32
    )

    for feature_index in range(features.shape[1]):
        resampled[:, feature_index] = np.interp(
            new_positions,
            old_positions,
            features[:, feature_index],
        )

    return resampled


def next_sequence_path(label):
    label_dir = SEQUENCE_DIR / label
    label_dir.mkdir(parents=True, exist_ok=True)

    numbers = []
    for path in label_dir.glob("sequence_*.npz"):
        try:
            numbers.append(int(path.stem.split("_")[-1]))
        except ValueError:
            pass

    next_number = max(numbers, default=0) + 1
    return label_dir / f"sequence_{next_number:04d}.npz"


def save_sequence(features, label):
    path = next_sequence_path(label)
    np.savez_compressed(
        path,
        features=features.astype(np.float32),
        label=np.asarray(label),
    )
    return path


def build_basic50_matches(annotations):
    target_lookup = {normalize_word(word): word for word in BASIC_WORDS}
    matches = []

    for index, annotation in enumerate(annotations):
        text = annotation.get("text", "")
        if normalize_word(text) in target_lookup:
            matches.append((index, annotation))

    return matches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", choices=["train", "val", "test"], default="train"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum NEW successful samples. 0 = all remaining.",
    )
    args = parser.parse_args()

    annotations = load_annotations(args.split)
    state = load_state()
    matches = build_basic50_matches(annotations)

    processed = set(state[args.split]["processed"])
    remaining = [
        (index, annotation)
        for index, annotation in matches
        if index not in processed
    ]

    print(f"MS-ASL {args.split}")
    print(f"Total annotations: {len(annotations):,}")
    print(f"Basic-50 matches: {len(matches):,}")
    print(f"Already processed: {len(processed):,}")
    print(f"Remaining: {len(remaining):,}")

    target_count = len(remaining) if args.limit == 0 else args.limit
    successful = 0

    for index, annotation in remaining:
        if successful >= target_count:
            break

        url = annotation.get("url")
        start_time = float(annotation.get("start_time", 0))
        end_time = float(annotation.get("end_time", 0))
        text = annotation.get("text", "")
        label = normalize_label(text)

        print(f"\n[{successful + 1}/{target_count}] index={index} word={label}")

        try:
            if not url:
                raise RuntimeError("Annotation has no video URL.")

            video_path = download_video(url)
            features = extract_sequence(video_path, start_time, end_time)
            output_path = save_sequence(features, label)

            state[args.split]["processed"].append(index)
            state[args.split]["failed"].pop(str(index), None)
            save_state(state)

            successful += 1
            print(f"Saved: {output_path}")
            print(f"Shape: {features.shape}")

        except Exception as exc:
            print(f"FAILED: {exc}")
            state[args.split]["failed"][str(index)] = {
                "error": str(exc),
                "word": text,
            }
            save_state(state)

    print("\n" + "=" * 60)
    print("DONE")
    print(f"Successful this run: {successful}")
    print(f"Total successful: {len(state[args.split]['processed'])}")
    print(f"State file: {STATE_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
