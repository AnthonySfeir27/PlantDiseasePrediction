"""Data model for one prediction result."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionResult:
    """Represents the output shown after analyzing one image."""

    disease_name: str
    confidence: float
    severity: str
    description: str
    recommendation: str
    is_real_model: bool
