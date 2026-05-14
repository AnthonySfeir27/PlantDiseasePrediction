"""Shared application configuration."""

APP_TITLE = "Plant Disease Prediction"
APP_SUBTITLE = "CNN-based leaf disease detection interface"
SUPPORTED_IMAGE_TYPES = ["jpg", "jpeg", "png"]

DISEASE_CLASSES = [
    "Tomato Healthy",
    "Tomato Early Blight",
    "Tomato Late Blight",
    "Tomato Leaf Mold",
]

DISEASE_DETAILS = {
    "Tomato Healthy": {
        "severity": "None",
        "description": "The uploaded leaf appears healthy in this demo prediction.",
        "recommendation": "Continue normal monitoring, watering, and sunlight exposure.",
    },
    "Tomato Early Blight": {
        "severity": "Medium",
        "description": "Early blight usually appears as dark spots with circular rings on older leaves.",
        "recommendation": "Remove infected leaves, improve air circulation, and avoid watering leaves directly.",
    },
    "Tomato Late Blight": {
        "severity": "High",
        "description": "Late blight can spread quickly and may damage leaves, stems, and fruit.",
        "recommendation": "Isolate affected plants, remove damaged leaves, and apply suitable treatment early.",
    },
    "Tomato Leaf Mold": {
        "severity": "Medium",
        "description": "Leaf mold often appears in humid conditions and affects leaf surfaces.",
        "recommendation": "Reduce humidity, increase ventilation, and prune crowded plant growth.",
    },
}
