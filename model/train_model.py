from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

import sys

# Allow running this file directly while still importing shared project utilities.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.paths import LANDMARKS_CSV_PATH, MODEL_PATH


def main() -> None:
    if not LANDMARKS_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Landmark CSV not found: {LANDMARKS_CSV_PATH}. Run scripts/extract_landmarks.py first."
        )

    df = pd.read_csv(LANDMARKS_CSV_PATH)
    if "label" not in df.columns:
        raise ValueError("Expected a 'label' column in landmark CSV.")

    X = df.drop(columns=["label"])
    y = df["label"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    # Pipeline keeps preprocessing and model inference consistent during deployment.
    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", RandomForestClassifier(n_estimators=300, random_state=42)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": pipeline,
            "label_encoder": label_encoder,
            "feature_columns": list(X.columns),
        },
        MODEL_PATH,
    )

    print(f"Saved trained model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()
