"""Real TensorFlow prediction service used by the Streamlit app."""

from functools import lru_cache

import numpy as np
import tensorflow as tf
from PIL import Image

from src.config import CLASS_NAMES_PATH, DISEASE_DETAILS, DISPLAY_NAMES, IMAGE_SIZE, MODEL_PATH
from src.models.model_status import ModelStatus
from src.models.prediction_result import PredictionResult
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
    class_index = int(np.argmax(probabilities))
    confidence = float(probabilities[class_index] * 100)

    raw_class_name = class_names[class_index]
    disease_name = DISPLAY_NAMES.get(raw_class_name, raw_class_name.replace("___", " ").replace("_", " "))
    details = DISEASE_DETAILS.get(
        disease_name,
        {
            "severity": "Unknown",
            "description": "The model returned a class that has no manual description yet.",
            "recommendation": "Review the image and verify the class manually.",
        },
    )

    return PredictionResult(
        disease_name=disease_name,
        confidence=confidence,
        severity=details["severity"],
        description=details["description"],
        recommendation=details["recommendation"],
        is_real_model=True,
    )


def prepare_image_for_model(image: Image.Image) -> np.ndarray:
    """Resize and batch one image for Keras inference."""
    resized_image = image.convert("RGB").resize(IMAGE_SIZE)
    image_array = tf.keras.utils.img_to_array(resized_image)
    return np.expand_dims(image_array, axis=0)
