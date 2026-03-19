from pathlib import Path
from typing import List

import cv2
import pandas as pd

import sys

# Allow running this file directly while still importing shared project utilities.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.mediapipe_helper import create_hands_detector, extract_21_hand_landmarks
from utils.paths import LANDMARKS_CSV_PATH, OUTPUT_CSV_PATH


def get_landmark_columns() -> List[str]:
    """Generate column names for 21 landmarks with x/y/z coordinates."""
    columns = []
    for idx in range(21):
        columns.extend([f"lm{idx}_x", f"lm{idx}_y", f"lm{idx}_z"])
    return columns


def main() -> None:
    if not OUTPUT_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Dataset index not found: {OUTPUT_CSV_PATH}. Run scripts/build_dataset.py first."
        )

    image_index_df = pd.read_csv(OUTPUT_CSV_PATH)
    landmark_columns = get_landmark_columns()

    records = []
    with create_hands_detector(static_image_mode=True, max_num_hands=1) as hands:
        for _, row in image_index_df.iterrows():
            image_path = Path(row["image_path"])
            label = row["label"]

            image = cv2.imread(str(image_path))
            if image is None:
                continue

            landmarks = extract_21_hand_landmarks(image, hands)
            if landmarks is None:
                continue

            sample = {"label": label}
            sample.update({col: float(val) for col, val in zip(landmark_columns, landmarks)})
            records.append(sample)

    landmarks_df = pd.DataFrame(records)
    LANDMARKS_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    landmarks_df.to_csv(LANDMARKS_CSV_PATH, index=False)

    print(f"Saved {len(landmarks_df)} landmark rows to: {LANDMARKS_CSV_PATH}")


if __name__ == "__main__":
    main()
