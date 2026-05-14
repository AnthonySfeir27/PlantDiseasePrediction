"""Demo prediction service used before a real model is trained."""

from PIL import Image

from src.models.prediction_result import PredictionCandidate, PredictionResult


def predict_leaf_disease_demo(image: Image.Image, filename: str) -> PredictionResult:
    """Return a deterministic fake prediction so the UI can be tested safely."""
    del image

    filename_lower = filename.lower()
    if "healthy" in filename_lower:
        disease_name = "Tomato Healthy"
        confidence = 91.0
        severity = "None"
        description = "Demo mode detected a healthy label from the file name."
        recommendation = "Train the model to replace demo predictions with real CNN results."
    elif "late" in filename_lower:
        disease_name = "Tomato Late Blight"
        confidence = 87.5
        severity = "High"
        description = "Demo mode selected late blight from the file name."
        recommendation = "Train the TensorFlow model before using this for real prediction."
    else:
        disease_name = "Tomato Early Blight"
        confidence = 76.2
        severity = "Medium"
        description = "Demo mode returns a sample disease result for interface testing."
        recommendation = "Use this only to test the GUI before training the model."

    top_predictions = [
        PredictionCandidate(disease_name=disease_name, confidence=confidence),
        PredictionCandidate(disease_name="Tomato Leaf Mold", confidence=12.8),
        PredictionCandidate(disease_name="Tomato Healthy", confidence=5.9),
    ]

    return PredictionResult(
        disease_name=disease_name,
        confidence=confidence,
        severity=severity,
        description=description,
        recommendation=recommendation,
        is_real_model=False,
        top_predictions=top_predictions,
    )
