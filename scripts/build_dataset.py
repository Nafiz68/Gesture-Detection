import csv
import shutil
from pathlib import Path
from typing import Dict, List

import kagglehub

import sys

# Allow running this file directly while still importing shared project utilities.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utils.paths import DATASET_DIR, OUTPUT_CSV_PATH, ensure_project_directories

DATASET_SLUG = "gti-upm/leapgestrecog"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def _has_participant_folders(path: Path) -> bool:
    """Return True if folder contains participant directories like 00..09."""
    return any(child.is_dir() and child.name.isdigit() for child in path.iterdir())


def _resolve_dataset_content_root(path: Path) -> Path:
    """Handle archives that extract either at root or inside a nested leapGestRecog folder."""
    if _has_participant_folders(path):
        return path

    nested = path / "leapGestRecog"
    if nested.exists() and nested.is_dir() and _has_participant_folders(nested):
        return nested

    return path


def download_leap_gesture_dataset() -> Path:
    """Download LeapGestRecog via kagglehub and mirror it into dataset/leapGestRecog."""
    source_path = _resolve_dataset_content_root(Path(kagglehub.dataset_download(DATASET_SLUG)))

    # Ensure our local project keeps a stable, predictable dataset location.
    DATASET_DIR.parent.mkdir(parents=True, exist_ok=True)
    if not DATASET_DIR.exists():
        DATASET_DIR.mkdir(parents=True, exist_ok=True)

    # Copy when participant folders are missing, even if placeholders like .gitkeep exist.
    if not _has_participant_folders(DATASET_DIR):
        shutil.copytree(source_path, DATASET_DIR, dirs_exist_ok=True)

    return DATASET_DIR


def scan_dataset_images(dataset_root: Path) -> List[Dict[str, str]]:
    """Scan participant folders and return image path + gesture label rows."""
    rows: List[Dict[str, str]] = []

    participant_dirs = sorted(
        [d for d in dataset_root.iterdir() if d.is_dir() and d.name.isdigit()]
    )

    for participant_dir in participant_dirs:
        for gesture_dir in sorted([d for d in participant_dir.iterdir() if d.is_dir()]):
            label = gesture_dir.name
            for image_path in sorted(gesture_dir.iterdir()):
                if image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    rows.append(
                        {
                            "image_path": str(image_path.resolve()),
                            "label": label,
                        }
                    )

    return rows


def write_index_csv(rows: List[Dict[str, str]], csv_path: Path) -> None:
    """Write indexed dataset rows to output/gesture_dataset.csv."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image_path", "label"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ensure_project_directories()

    dataset_root = download_leap_gesture_dataset()
    rows = scan_dataset_images(dataset_root)
    write_index_csv(rows, OUTPUT_CSV_PATH)

    print(f"Indexed {len(rows)} images into: {OUTPUT_CSV_PATH}")


if __name__ == "__main__":
    main()
