"""Real TensorFlow prediction service used by the Streamlit app."""

from functools import lru_cache

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import (
    CLASS_NAMES_PATH,
    DISEASE_DETAILS,
    DISPLAY_NAMES,
    IMAGE_SIZE,
    MODEL_PATH,
    TOP_PREDICTION_COUNT,
)
from src.models.model_status import ModelStatus
from src.models.prediction_result import PredictionCandidate, PredictionResult
from src.utils.file_io import read_json


def get_model_status() -> ModelStatus:
    """Return whether the trained model and metadata are available."""
    if not MODEL_PATH.exists():
        return ModelStatus(False, "Model file not found. Train the model first with: python train.py")
    if not CLASS_NAMES_PATH.exists():
        return ModelStatus(False, "Class metadata not found. Re-run training to generate class_names.json.")
    return ModelStatus(True, "Real TensorFlow model loaded for predictions.")


@lru_cache(maxsize=1)
def load_model() -> tf.keras.Model:
    """Load the trained Keras model once per Streamlit process."""
    return tf.keras.models.load_model(MODEL_PATH)


@lru_cache(maxsize=1)
def load_class_names() -> list[str]:
    """Load class names saved during training."""
    return read_json(CLASS_NAMES_PATH)


def predict_leaf_disease(image: Image.Image) -> PredictionResult:
    """Predict the plant disease class for one PIL image."""
    model = load_model()
    class_names = load_class_names()

    model_input = prepare_image_for_model(image)
    probabilities = model.predict(model_input, verbose=0)[0]
    top_predictions = build_top_predictions(probabilities, class_names)
    best_prediction = top_predictions[0]
    details = get_disease_details(best_prediction.disease_name)

    return PredictionResult(
        disease_name=best_prediction.disease_name,
        confidence=best_prediction.confidence,
        severity=details["severity"],
        description=details["description"],
        recommendation=details["recommendation"],
        is_real_model=True,
        top_predictions=top_predictions,
    )


def build_top_predictions(probabilities: np.ndarray, class_names: list[str]) -> list[PredictionCandidate]:
    """Return the highest probability model classes in descending order."""
    count = min(TOP_PREDICTION_COUNT, len(class_names))
    sorted_indexes = np.argsort(probabilities)[::-1][:count]

    predictions = []
    for class_index in sorted_indexes:
        raw_class_name = class_names[int(class_index)]
        disease_name = format_class_name(raw_class_name)
        confidence = float(probabilities[int(class_index)] * 100)
        predictions.append(PredictionCandidate(disease_name=disease_name, confidence=confidence))

    return predictions


def format_class_name(raw_class_name: str) -> str:
    """Convert a raw dataset folder name into a readable disease name."""
    if raw_class_name in DISPLAY_NAMES:
        return DISPLAY_NAMES[raw_class_name]
    return raw_class_name.replace("___", " ").replace("__", " ").replace("_", " ").strip()


def get_disease_details(disease_name: str) -> dict[str, str]:
    """Return manual display details for a disease class."""
    return DISEASE_DETAILS.get(
        disease_name,
        {
            "severity": "Unknown",
            "description": "The model returned a class that has no manual description yet.",
            "recommendation": "Review the image and verify the class manually.",
        },
    )


def prepare_image_for_model(image: Image.Image) -> np.ndarray:
    """Resize and batch one image for Keras inference."""
    resized_image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(resized_image)
    return np.expand_dims(image_array, axis=0)
