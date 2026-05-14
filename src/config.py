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
TOP_PREDICTION_COUNT = 3

REQUIRED_DATASET_CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
]

DISPLAY_NAMES = {
    "Pepper__bell___Bacterial_spot": "Bell Pepper Bacterial Spot",
    "Pepper__bell___healthy": "Bell Pepper Healthy",
    "Potato___Early_blight": "Potato Early Blight",
    "Potato___Late_blight": "Potato Late Blight",
    "Potato___healthy": "Potato Healthy",
    "Tomato_Bacterial_spot": "Tomato Bacterial Spot",
    "Tomato_Early_blight": "Tomato Early Blight",
    "Tomato_Late_blight": "Tomato Late Blight",
    "Tomato_Leaf_Mold": "Tomato Leaf Mold",
    "Tomato_Septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite": "Tomato Spider Mites",
    "Tomato__Target_Spot": "Tomato Target Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus": "Tomato Yellow Leaf Curl Virus",
    "Tomato__Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato_healthy": "Tomato Healthy",
    "Tomato___healthy": "Tomato Healthy",
    "Tomato___Early_blight": "Tomato Early Blight",
    "Tomato___Late_blight": "Tomato Late Blight",
    "Tomato___Leaf_Mold": "Tomato Leaf Mold",
}

DISEASE_DETAILS = {
    "Bell Pepper Bacterial Spot": {
        "severity": "Medium",
        "description": "Bacterial spot can create dark lesions on pepper leaves and may reduce plant strength.",
        "recommendation": "Remove infected leaves, avoid overhead watering, and improve air circulation.",
    },
    "Bell Pepper Healthy": {
        "severity": "None",
        "description": "The pepper leaf is classified as healthy by the model.",
        "recommendation": "Continue normal watering, sunlight exposure, and regular monitoring.",
    },
    "Potato Early Blight": {
        "severity": "Medium",
        "description": "Early blight commonly causes brown spots and yellowing on older potato leaves.",
        "recommendation": "Remove infected foliage, avoid wet leaves, and monitor nearby plants.",
    },
    "Potato Late Blight": {
        "severity": "High",
        "description": "Late blight can spread quickly and severely damage potato plants.",
        "recommendation": "Separate affected plants, remove damaged leaves, and apply suitable treatment quickly.",
    },
    "Potato Healthy": {
        "severity": "None",
        "description": "The potato leaf is classified as healthy by the model.",
        "recommendation": "Continue normal care and inspect leaves regularly.",
    },
    "Tomato Bacterial Spot": {
        "severity": "Medium",
        "description": "Bacterial spot creates dark, water-soaked marks and can weaken tomato leaves.",
        "recommendation": "Remove affected leaves, avoid splashing water, and improve spacing between plants.",
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
    "Tomato Septoria Leaf Spot": {
        "severity": "Medium",
        "description": "Septoria leaf spot often appears as many small circular spots on tomato leaves.",
        "recommendation": "Remove lower infected leaves, keep soil from splashing, and increase plant spacing.",
    },
    "Tomato Spider Mites": {
        "severity": "Medium",
        "description": "Spider mite damage may appear as speckled leaves, webbing, or leaf discoloration.",
        "recommendation": "Inspect leaf undersides, isolate the plant, and use an appropriate mite control method.",
    },
    "Tomato Target Spot": {
        "severity": "Medium",
        "description": "Target spot creates circular brown lesions that may expand and damage tomato foliage.",
        "recommendation": "Remove affected leaves, reduce leaf wetness, and improve airflow.",
    },
    "Tomato Yellow Leaf Curl Virus": {
        "severity": "High",
        "description": "Yellow leaf curl virus can cause curling, yellowing, and reduced tomato growth.",
        "recommendation": "Control whiteflies, remove severely infected plants, and avoid using infected plant material.",
    },
    "Tomato Mosaic Virus": {
        "severity": "High",
        "description": "Tomato mosaic virus can create mottled coloring, distorted growth, and weaker plants.",
        "recommendation": "Remove infected plants, disinfect tools, and avoid handling healthy plants after infected ones.",
    },
    "Tomato Healthy": {
        "severity": "None",
        "description": "The leaf is classified as healthy. No visible disease pattern was detected by the model.",
        "recommendation": "Continue normal watering, sunlight exposure, and regular plant monitoring.",
    },
}
