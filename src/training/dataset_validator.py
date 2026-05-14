"""Dataset validation before training."""

from pathlib import Path

from src.config import REQUIRED_DATASET_CLASSES

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def validate_dataset(dataset_dir: Path) -> None:
    """Stop training early if the required class folders are missing or empty."""
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder does not exist: {dataset_dir}")

    missing_classes = []
    empty_classes = []

    for class_name in REQUIRED_DATASET_CLASSES:
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            missing_classes.append(class_name)
            continue

        image_count = count_images(class_dir)
        if image_count == 0:
            empty_classes.append(class_name)

    if missing_classes or empty_classes:
        message_parts = []
        if missing_classes:
            message_parts.append("Missing folders: " + ", ".join(missing_classes))
        if empty_classes:
            message_parts.append("Empty folders: " + ", ".join(empty_classes))
        raise ValueError("Invalid dataset. " + " | ".join(message_parts))


def count_images(folder: Path) -> int:
    """Count supported image files inside one class folder."""
    return sum(1 for path in folder.rglob("*") if path.suffix.lower() in SUPPORTED_EXTENSIONS)


def get_dataset_summary(dataset_dir: Path) -> dict[str, int]:
    """Return image counts for each required class."""
    return {
        class_name: count_images(dataset_dir / class_name)
        for class_name in REQUIRED_DATASET_CLASSES
        if (dataset_dir / class_name).exists()
    }
