from pathlib import Path
from typing import Any, Optional
from urllib.request import urlopen

import cv2
import mediapipe as mp
import numpy as np


HAND_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)
HAND_LANDMARKER_MODEL_PATH = (
    Path(__file__).resolve().parents[1] / "model" / "hand_landmarker.task"
)

LEGACY_SOLUTIONS_AVAILABLE = hasattr(mp, "solutions")

if LEGACY_SOLUTIONS_AVAILABLE:
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
else:
    mp_hands = None
    mp_drawing = None
    mp_tasks = mp.tasks
    mp_vision = mp_tasks.vision


def _ensure_hand_landmarker_model() -> Path:
    """Download a default hand landmarker model once for Tasks-based MediaPipe."""
    HAND_LANDMARKER_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if HAND_LANDMARKER_MODEL_PATH.exists() and HAND_LANDMARKER_MODEL_PATH.stat().st_size > 0:
        return HAND_LANDMARKER_MODEL_PATH

    with urlopen(HAND_LANDMARKER_MODEL_URL, timeout=60) as response:
        model_bytes = response.read()

    HAND_LANDMARKER_MODEL_PATH.write_bytes(model_bytes)
    return HAND_LANDMARKER_MODEL_PATH


class _TasksHandsWrapper:
    """Minimal wrapper to mimic the Hands lifecycle used by the scripts."""

    def __init__(self, hand_landmarker: Any):
        self._hand_landmarker = hand_landmarker

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False

    def close(self) -> None:
        self._hand_landmarker.close()

    def detect(self, image_bgr: np.ndarray):
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
        return self._hand_landmarker.detect(mp_image)


def create_hands_detector(
    static_image_mode: bool = False,
    max_num_hands: int = 1,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> Any:
    """Initialize and return a reusable MediaPipe Hands detector."""
    if LEGACY_SOLUTIONS_AVAILABLE:
        return mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    model_path = _ensure_hand_landmarker_model()
    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.IMAGE,
        num_hands=max_num_hands,
        min_hand_detection_confidence=min_detection_confidence,
        min_hand_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )
    return _TasksHandsWrapper(mp_vision.HandLandmarker.create_from_options(options))


def process_image(image_bgr: np.ndarray, hands_detector: Any):
    """Run MediaPipe Hands on a BGR image and return the raw result object."""
    if LEGACY_SOLUTIONS_AVAILABLE:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        return hands_detector.process(image_rgb)
    return hands_detector.detect(image_bgr)


def extract_21_hand_landmarks(
    image_bgr: np.ndarray, hands_detector: Any
) -> Optional[np.ndarray]:
    """Return flattened 21x3 hand landmarks (63 values) or None if no hand is found."""
    results = process_image(image_bgr, hands_detector)

    if LEGACY_SOLUTIONS_AVAILABLE:
        if not results.multi_hand_landmarks:
            return None
        first_hand = results.multi_hand_landmarks[0]
    else:
        if not results.hand_landmarks:
            return None
        first_hand = results.hand_landmarks[0]

    landmarks_iter = first_hand.landmark if LEGACY_SOLUTIONS_AVAILABLE else first_hand

    flattened = []
    for landmark in landmarks_iter:
        flattened.extend([landmark.x, landmark.y, landmark.z])

    return np.array(flattened, dtype=np.float32)


def draw_hand_landmarks(image_bgr: np.ndarray, results) -> np.ndarray:
    """Draw detected hand landmarks on a copy of the image for visualization."""
    output = image_bgr.copy()

    if LEGACY_SOLUTIONS_AVAILABLE:
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(output, hand_landmarks, mp_hands.HAND_CONNECTIONS)
        return output

    if not results.hand_landmarks:
        return output

    height, width = output.shape[:2]
    for hand_landmarks in results.hand_landmarks:
        points = []
        for landmark in hand_landmarks:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            cv2.circle(output, (x, y), 3, (0, 255, 0), -1)

        for connection in mp_vision.HandLandmarksConnections.HAND_CONNECTIONS:
            start = connection.start
            end = connection.end
            cv2.line(output, points[start], points[end], (255, 0, 0), 2)

    return output
