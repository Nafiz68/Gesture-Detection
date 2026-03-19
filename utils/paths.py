from pathlib import Path

# Centralized project paths so all scripts import the same locations.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "dataset" / "leapGestRecog"
OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_CSV_PATH = OUTPUT_DIR / "gesture_dataset.csv"
LANDMARKS_CSV_PATH = OUTPUT_DIR / "gesture_landmarks.csv"
MODEL_DIR = PROJECT_ROOT / "model"
MODEL_PATH = MODEL_DIR / "gesture_model.pkl"


def ensure_project_directories() -> None:
    """Create required runtime directories if they do not already exist."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
