"""Model availability status shown in the interface."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelStatus:
    """Describes whether the trained TensorFlow model is available."""

    is_ready: bool
    message: str
