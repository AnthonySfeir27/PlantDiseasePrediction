"""Print dataset readiness information before training."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import DATASET_DIR
from src.training.dataset_validator import get_dataset_summary, validate_dataset


def main() -> None:
    """Validate required folders and print image counts."""
    validate_dataset(DATASET_DIR)
    summary = get_dataset_summary(DATASET_DIR)

    print("Dataset is ready for training.\n")
    for class_name, image_count in summary.items():
        print(f"{class_name}: {image_count} images")


if __name__ == "__main__":
    main()
