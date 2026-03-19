# Gesture Detection (ML Training Pipeline)

This project builds a hand gesture recognition model using the LeapGestRecog dataset, MediaPipe hand landmarks, and a scikit-learn classifier.

## Features

- Dataset indexing from image folders
- Hand landmark extraction (21 points x 3 coordinates = 63 features)
- Model training with preprocessing pipeline
- Real-time webcam inference with predicted label and confidence
- MediaPipe compatibility for newer Tasks API environments

## Project Structure

```text
gesture-ml-training/
  requirements.txt
  scripts/
    build_dataset.py
    extract_landmarks.py
  model/
    train_model.py
  test/
    webcam_test.py
  utils/
    mediapipe_helper.py
    paths.py
```

## Requirements

- Python 3.10+ (tested with newer Python and MediaPipe Tasks API)
- Webcam (for live testing)

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset

Expected dataset location:

```text
dataset/leapGestRecog/
```

The folder should contain participant subfolders (`00`, `01`, ..., `09`) and gesture class folders inside each.

If your dataset is already present, continue to pipeline steps.

## Pipeline (Run in Order)

### 1) Build dataset index CSV

Scans image files and writes:

- `output/gesture_dataset.csv`

```powershell
python scripts/build_dataset.py
```

### 2) Extract hand landmarks

Reads image index and writes:

- `output/gesture_landmarks.csv`

```powershell
python scripts/extract_landmarks.py
```

### 3) Train model

Trains classifier and writes:

- `model/gesture_model.pkl`

```powershell
python model/train_model.py
```

## Real-time Webcam Test

Runs inference using webcam feed:

```powershell
python test/webcam_test.py
```

What you should see:

- Hand landmarks drawn on frame
- Predicted class label (example: `07_ok`)
- Confidence percentage (example: `94.2%`)

Press `q` to quit.

## Label Classes

The model predicts these dataset class labels:

1. `01_palm`
2. `02_l`
3. `03_fist`
4. `04_fist_moved`
5. `05_thumb`
6. `06_index`
7. `07_ok`
8. `08_palm_moved`
9. `09_c`
10. `10_down`

## Notes

- `dataset/`, `output/`, virtual environment files, and large model artifacts are ignored by git.
- For best webcam results: good lighting, visible full hand, simple background, and stable gestures.
