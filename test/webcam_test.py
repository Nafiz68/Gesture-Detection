import cv2
import joblib
import pandas as pd

import sys
from pathlib import Path

# Allow running this file directly while still importing shared project utilities.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.mediapipe_helper import (
    create_hands_detector,
    draw_hand_landmarks,
    extract_21_hand_landmarks,
    process_image,
)
from utils.paths import MODEL_PATH


def main() -> None:
    if not MODEL_PATH.exists() or MODEL_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run model/train_model.py first."
        )

    model_bundle = joblib.load(MODEL_PATH)
    pipeline = model_bundle["pipeline"]
    label_encoder = model_bundle["label_encoder"]
    feature_columns = model_bundle["feature_columns"]

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Unable to open webcam.")

    with create_hands_detector(static_image_mode=False, max_num_hands=1) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            results = process_image(frame, hands)
            landmarks = extract_21_hand_landmarks(frame, hands)

            if landmarks is not None:
                feature_df = pd.DataFrame([landmarks], columns=feature_columns)
                pred = pipeline.predict(feature_df)[0]
                label = label_encoder.inverse_transform([pred])[0]
                confidence_text = ""
                if hasattr(pipeline, "predict_proba"):
                    confidence = float(pipeline.predict_proba(feature_df)[0].max())
                    confidence_text = f" ({confidence * 100:.1f}%)"
                cv2.putText(
                    frame,
                    f"Gesture: {label}{confidence_text}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            frame = draw_hand_landmarks(frame, results)
            cv2.imshow("Gesture Recognition Test", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
