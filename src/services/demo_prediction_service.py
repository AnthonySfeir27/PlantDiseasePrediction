"""Temporary prediction service used before connecting the trained CNN model.

This file intentionally keeps the same service boundary that the real model will use.
Later, only this service needs to change; the UI can stay mostly untouched.
"""

from hashlib import sha256

from src.config import DISEASE_CLASSES, DISEASE_DETAILS
from src.models.prediction_result import PredictionResult


def predict_leaf_disease_demo(file_name: str) -> PredictionResult:
    """Return a deterministic demo prediction based on the uploaded file name.

    This makes the interface testable before the CNN model exists.
    It is not real AI yet. We will replace it with TensorFlow inference next.
    """
    file_hash = sha256(file_name.encode("utf-8")).hexdigest()
    class_index = int(file_hash[:2], 16) % len(DISEASE_CLASSES)
    confidence = 72.0 + (int(file_hash[2:4], 16) % 25)

    disease_name = DISEASE_CLASSES[class_index]
    details = DISEASE_DETAILS[disease_name]

    return PredictionResult(
        disease_name=disease_name,
        confidence=confidence,
        severity=details["severity"],
        description=details["description"],
        recommendation=details["recommendation"],
    )
