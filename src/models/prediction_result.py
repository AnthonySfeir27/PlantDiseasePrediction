"""Data models for prediction output shown in the UI."""

from dataclasses import dataclass


@dataclass(frozen=True)
class PredictionCandidate:
    """Represents one class candidate returned by the model."""

    disease_name: str
    confidence: float


@dataclass(frozen=True)
class PredictionResult:
    """Represents the complete output shown after analyzing one image."""

    disease_name: str
    confidence: float
    severity: str
    description: str
    recommendation: str
    is_real_model: bool
    top_predictions: list[PredictionCandidate]
