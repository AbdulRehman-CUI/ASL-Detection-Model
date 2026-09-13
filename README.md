# Sign Language Translator: Phases 1, 2 & 2B

This project implements a hand-tracking pipeline and image-to-landmark dataset system for static sign recognition.

**Current Status:** Phase 4 real-time inference implemented for digits `0`-`9` and letters `A`-`Z`. Phase 5 sentence building and speech output is next.

## Architecture Overview

The complete pipeline will be:

```
Webcam
  ↓
OpenCV (frame capture)
  ↓
MediaPipe (hand detection)
  ↓
21 hand landmarks
  ↓
Feature extraction & normalization (landmark_utils.py)
  ↓
ML classifier training (Phase 3)
  ↓
Sign prediction & filtering
  ↓
Text/Speech output
```

## Project Structure

```
project/
├── phase1_hand_tracking.py       # Real-time hand landmark visualization
├── phase2_data_collection.py     # Collect labeled landmark data
├── phase2b_image_to_landmarks.py  # Convert asl_dataset images to landmark data
├── phase3_train_classifier.py     # Train and save a landmark classifier
├── phase4_realtime_inference.py   # Real-time sign prediction from webcam
├── utils/
│   └── landmark_utils.py         # Reusable landmark processing functions
├── data/
│   └── sign_landmarks.csv        # Collected landmark dataset
├── models/
│   ├── sign_classifier.joblib    # Trained classifier
│   └── labels.json               # Model label map
├── sign_language_translator_project_context.md
├── requirements.txt
└── README.md
```

## Install

From the project directory, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

**Notes:**
- MediaPipe is pinned to `0.10.21` for the `mp.solutions.hands` API.
- NumPy is used for landmark normalization.
- Python 3.11.9 is recommended.

## Phase 1: Hand Tracking

Demonstrates the hand-tracking pipeline: `Webcam → OpenCV → MediaPipe → Landmarks → Display`

### Run Phase 1

```powershell
python phase1_hand_tracking.py
```

- The webcam opens in a window titled `Sign Language Translator - Phase 1`.
- Put one or two hands in view to see the 21 landmarks and connections.
- Press `q` or `Esc` to exit.

**Success criteria:** Live feed opens, landmarks follow hand in real time, exits cleanly.

---

## Phase 2: Data Collection

Collects labeled landmark data for training a sign language classifier.

### Overview

Phase 2 allows you to:

1. Select a sign/label (from a predefined vocabulary).
2. Open the webcam.
3. Perform the sign in front of the camera.
4. Let MediaPipe detect the hand and extract landmarks.
5. Save the landmark features with the label to a CSV dataset.
6. Repeat for many samples of the same sign.
7. Collect multiple signs into a single dataset.

### Vocabulary

The initial vocabulary includes these signs (easily customizable):

- `HELLO`
- `THANK_YOU`
- `YES`
- `NO`
- `HELP`
- `WATER`
- `STOP`
- `PLEASE`
- `I_LOVE_YOU`

### Run Phase 2

```powershell
python phase2_data_collection.py
```

1. A menu displays the available signs. Enter the number of the sign you want to collect.
2. The webcam opens. The interface shows:
   - Current sign label
   - Sample count (e.g., `Samples: 42 / 300`)
   - Collection status (COLLECTING or PAUSED)
   - Hand detection status (YES or NO)
   - Visual hand landmarks overlaid on the video

3. **Controls:**
   - `SPACE` — Start/pause collection
   - `C` — Change to a different sign
   - `R` — Reset the sample counter for the current sign (data is not deleted)
   - `Q` or `ESC` — Quit

4. Collect samples by:
   - Pressing `SPACE` to start collection.
   - Performing the sign in front of the camera while keeping your hand visible.
   - The program automatically captures samples at ~3.3 FPS to avoid duplicates.
   - Continue until the target (300 samples by default) is reached.

5. When the target is reached, you can:
   - Change to another sign by pressing `C`.
   - Collect more samples for the current sign.
   - Quit by pressing `Q`.

### Data Collection Tips

For best results:

- **Position variety:** Perform the sign in different areas of the frame (left, center, right, top, bottom).
- **Distance variation:** Hold your hand at slightly different distances from the camera.
- **Rotation:** Perform the sign with subtle rotations and tilts.
- **Natural variation:** Allow slight natural variation in how you perform the sign each time.
- **Lighting:** Collect samples in different lighting if possible.
- **Background:** Vary the background behind your hand where practical.

The system normalizes landmarks relative to wrist position and scale, so absolute position/distance matters less than variation in hand shape and configuration.

### Dataset Format

The collected data is saved to `data/sign_landmarks.csv`. Each row contains:

- **Columns 1–63:** Normalized hand landmark coordinates (x₁, y₁, z₁, x₂, y₂, z₂, ..., x₂₁, y₂₁, z₂₁)
- **Column 64:** Sign label (e.g., `HELLO`, `YES`)

Example:

```csv
feature_0,feature_1,...,feature_62,label
-0.0123,0.0456,...,0.2891,HELLO
-0.0089,0.0512,...,0.2756,HELLO
0.0234,0.0389,...,0.2945,YES
...
```

### Feature Normalization

The features are normalized to be invariant to hand position and scale:

1. **Translation:** Landmarks are translated so the wrist (landmark 0) is at the origin.
2. **Scale:** Landmarks are divided by the distance from wrist to middle finger MCP (landmark 9).

This ensures that:
- The same sign performed in different parts of the frame produces similar features.
- Hand distance from the camera has less impact.
- The model learns hand shape/configuration rather than absolute position.

The normalization is implemented in `utils/landmark_utils.py` and must be applied identically during training (Phase 3) and inference (Phase 4).

### Sampling Strategy

To avoid collecting thousands of nearly-identical frames:

- Samples are captured approximately every 10 frames (~0.33 seconds at 30 FPS).
- Only samples where a hand is successfully detected are saved.
- You control overall collection rate by pressing `SPACE` to pause/resume.

### Dataset Integrity

Before each sample is saved, the system verifies:

- A hand was detected.
- Exactly 21 landmarks were extracted.
- The feature vector has 63 values.
- The label is valid.

Samples with missing or invalid data are silently skipped.

### Phase 2B: Convert Ready-Made Images

Manual webcam collection is optional. To convert the class-folder image dataset in `asl_dataset/`:

```powershell
python phase2b_image_to_landmarks.py
```

The script walks the `asl_dataset/` class folders, runs MediaPipe on each image,
normalizes detected landmarks, and writes valid rows to `data/sign_landmarks_from_images.csv`.
Images where MediaPipe cannot detect a hand are skipped. The output has 63 features plus
one label from the canonical static-sign set: digits `0` through `9` and letters `A`
through `Z`.

The converter recreates the CSV at the start of each run, so rerunning it does not duplicate
previous samples. The older `X.npy`/`Y.npy` digit source is not used by this converter,
which avoids mixing labels such as `ZERO`/`ONE` with `0`/`1`.

---

## Phase 3: Classifier Training

Trains a Random Forest classifier on `data/sign_landmarks_from_images.csv` and saves:

- `models/sign_classifier.joblib`
- `models/labels.json`

### Run Phase 3

```powershell
python phase3_train_classifier.py
```

Review the printed classification report before moving to live inference.

---

## Phase 4: Real-Time Inference

Loads the trained model and predicts static signs from webcam hand landmarks.

### Run Phase 4

```powershell
python phase4_realtime_inference.py
```

The webcam window shows:

- Hand detection status
- Current frame prediction
- Prediction confidence
- Stable prediction after repeated matching frames

Controls:

- `Q` or `ESC` - quit
- `R` - reset the stable prediction

Phase 4 recognizes the current static dataset labels: digits `0` through `9` and letters `A` through `Z`.

### Run the Browser Frontend

The React frontend uses the browser camera and sends frames to the local prediction
service. Start the service from the project root, then start Vite in a second terminal:

```powershell
python prediction_server.py
cd Frontend
npm run dev
```

Open the Vite URL, allow camera access, and click **Start Recognition**. Static mode
classifies each frame; dynamic mode continuously refreshes its 30-frame sequence window.

---

## Common Errors

### Missing Dependencies

**Error:** `ModuleNotFoundError: No module named 'cv2'` or `mediapipe`

**Solution:**
1. Activate the virtual environment: `.\.venv\Scripts\Activate.ps1`
2. Run: `python -m pip install -r requirements.txt`
3. Verify: `python -c "import cv2; import mediapipe; print('OK')"`

### Could Not Open Webcam

**Error:** `Could not open webcam at camera index 0`

**Solution:**
- Close other applications using the camera (e.g., Teams, OBS, other video apps).
- Check operating-system camera permissions.
- Connect a webcam if using a laptop without a built-in camera.
- Try changing `CAMERA_INDEX` in the script if you have multiple cameras.

### PowerShell Blocks Activation

**Error:** `cannot be loaded because running scripts is disabled on this system`

**Solution:**
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Dataset File Errors

**Error:** `Error saving sample` or permission errors

**Solution:**
- Ensure the `data/` directory is writable.
- Check that `data/sign_landmarks.csv` is not open in Excel or another application.
- Close and reopen the file if it is.

---

## Verification

### After Phase 2 Completion

Verify the dataset is correct:

1. **Check the CSV file:**
   ```powershell
   import csv
   with open('data/sign_landmarks.csv') as f:
       reader = csv.reader(f)
       header = next(reader)
       print(f"Columns: {len(header)}")  # Should be 64
       for i, row in enumerate(reader):
           if i < 3:
               print(f"Row {i}: {len(row)} values, label={row[-1]}")
   ```

2. **Expected output:**
   ```
   Columns: 64
   Row 0: 64 values, label=HELLO
   Row 1: 64 values, label=HELLO
   Row 2: 64 values, label=YES
   ```

3. **Row count by label:**
   ```powershell
   python -c "import csv; from collections import Counter; rows = list(csv.reader(open('data/sign_landmarks.csv'))); labels = [r[-1] for r in rows[1:]]; print(dict(Counter(labels)))"
   ```

---

## Next Steps

After Phase 2, you will have a dataset ready for Phase 3 (classifier training):

- The dataset contains many labeled examples of each sign's hand landmarks.
- Phase 3 will train an ML classifier (e.g., Random Forest, SVM) using this data.
- Phase 4 adds real-time inference and stability filtering.
- Phase 5 will add sentence building and speech output.

---

## Architecture Notes for Developers

### Reusable Components

- **`utils/landmark_utils.py`:** Provides reusable functions for:
  - Extracting landmarks from MediaPipe results
  - Normalizing landmarks (wrist-relative, scale-invariant)
  - Converting landmarks to feature vectors
  - Validating feature vectors
  
  These functions are used in Phase 2 (data collection) and will be reused in Phase 3 (training) and Phase 4 (inference).

- **Phase 1 (`phase1_hand_tracking.py`):** Remains unchanged. It provides a simple demonstration of the hand-tracking pipeline without modification.

### Why Landmark Features?

We use extracted landmark coordinates (not raw images) because:

1. **Compact:** 63 numbers per sample vs. thousands per image.
2. **Fast:** Smaller input, faster training and inference.
3. **Robust:** Invariant to lighting, background, camera angle (after normalization).
4. **Interpretable:** Direct geometric features without black-box neural networks.

### Why Normalization?

Normalization makes features position and scale invariant:

- Same sign at left edge ≈ same sign at center
- Same sign far from camera ≈ same sign close to camera
- Model learns hand configuration, not absolute position.

---

## License & Credits

This is a hackathon project for the Sign Language to Text/Speech Translator.

Phase 1 completed: Hand tracking with MediaPipe.
Phase 2 completed: Data collection pipeline.
Phase 3 completed: Classifier training and saved model artifact.
Phase 4 implemented: Real-time inference and stability filtering.
Phase 5 (pending): Sentence building and speech output.
