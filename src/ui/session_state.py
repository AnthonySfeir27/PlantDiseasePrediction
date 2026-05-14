"""Helpers for Streamlit session state."""

import streamlit as st

from src.models.prediction_result import PredictionResult


def initialize_session_state() -> None:
    """Create required session state keys if they do not exist."""
    if "prediction_history" not in st.session_state:
        st.session_state.prediction_history = []


def add_prediction_to_history(prediction: PredictionResult) -> None:
    """Store one prediction result in the current Streamlit session."""
    st.session_state.prediction_history.append(prediction)
