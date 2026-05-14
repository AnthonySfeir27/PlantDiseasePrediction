"""Fallback prediction service used before the CNN model is trained."""

from hashlib import sha256

from PIL import Image

from src.config import DISEASE_DETAILS, DISPLAY_NAMES, REQUIRED_DATASET_CLASSES
from src.models.prediction_result import PredictionResult


def predict_leaf_disease_demo(image: Image.Image, file_name: str) -> PredictionResult:
    """Return a deterministic placeholder prediction while the model is unavailable."""
    file_hash = sha256(f"{file_name}-{image.size}".encode("utf-8")).hexdigest()
    class_index = int(file_hash[:2], 16) % len(REQUIRED_DATASET_CLASSES)
    confidence = 70.0 + (int(file_hash[2:4], 16) % 26)

    raw_class_name = REQUIRED_DATASET_CLASSES[class_index]
    disease_name = DISPLAY_NAMES[raw_class_name]
    details = DISEASE_DETAILS[disease_name]

    return PredictionResult(
        disease_name=disease_name,
        confidence=confidence,
        severity=details["severity"],
        description=details["description"],
        recommendation=details["recommendation"],
        is_real_model=False,
    )
