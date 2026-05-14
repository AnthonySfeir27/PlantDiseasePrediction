"""Shared application configuration."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATASET_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "plant_disease_model.keras"
CLASS_NAMES_PATH = MODEL_DIR / "class_names.json"
TRAINING_HISTORY_PATH = MODEL_DIR / "training_history.json"

APP_TITLE = "Plant Disease Prediction"
APP_SUBTITLE = "CNN transfer-learning leaf disease detector"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.2
RANDOM_SEED = 42

REQUIRED_DATASET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
]

DISPLAY_NAMES = {
    "Tomato___healthy": "Tomato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
}

DISEASE_DETAILS = {
    "Tomato Healthy": {
        "severity": "None",
        "description": "The leaf is classified as healthy. No visible disease pattern was detected by the model.",
        "recommendation": "Continue normal watering, sunlight exposure, and regular plant monitoring.",
    },
    "Tomato Early Blight": {
        "severity": "Medium",
        "description": "Early blight commonly creates dark circular spots, usually starting on older tomato leaves.",
        "recommendation": "Remove infected leaves, avoid wetting foliage, improve airflow, and monitor nearby plants.",
    },
    "Tomato Late Blight": {
        "severity": "High",
        "description": "Late blight is aggressive and can spread quickly across leaves, stems, and fruit.",
        "recommendation": "Separate affected plants, remove damaged leaves, reduce humidity, and apply suitable treatment quickly.",
    },
    "Tomato Leaf Mold": {
        "severity": "Medium",
        "description": "Leaf mold is commonly linked to humid conditions and poor ventilation around tomato plants.",
        "recommendation": "Improve ventilation, reduce humidity, prune crowded leaves, and avoid overhead watering.",
    },
}
