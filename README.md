# Sign Language Recognition

This project recognizes American Sign Language using MediaPipe hand landmarks and machine-learning models.

## Workflow

The system captures a video frame, detects the hand, extracts and normalizes its 21 landmarks, and sends those features to a trained classifier. Static recognition classifies one hand shape at a time. Dynamic recognition analyzes a rolling sequence of frames to recognize movement-based signs and words.

## Architecture

```text
Camera
	-> OpenCV
	-> MediaPipe hand tracking
	-> Landmark normalization
	-> Static or dynamic classifier
	-> Sign prediction
```

## Phases

1. **Hand tracking:** Detect and display hand landmarks.
2. **Data preparation:** Collect or convert labeled hand-landmark samples.
3. **Model training:** Train static and dynamic classifiers.
4. **Recognition:** Run real-time static or dynamic predictions.
5. **Evaluation:** Test model accuracy and review classification results.

## Status

- **Static recognition:** Working and tested for hand shapes, letters, and numbers.
- **Dynamic recognition:** Working and tested for movement-based signs and words.
- **Uploaded model:** Only the static model is included because the dynamic model exceeds the repository file-size limit.

## Setup

```powershell
python -m pip install -r requirements.txt
```

## Run Recognition

Static signs:

```powershell
python phase4_realtime_inference.py
```

Dynamic signs:

```powershell
python phase4_dynamic_realtime_inference.py
```

The system uses OpenCV for video capture, MediaPipe for hand tracking, normalized hand landmarks for features, and trained classifiers for prediction.
