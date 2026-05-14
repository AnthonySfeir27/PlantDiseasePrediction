"""UI rendering for uploaded images and prediction results."""

from PIL import Image
import streamlit as st

from src.models.prediction_result import PredictionResult


def render_uploaded_image(uploaded_file) -> Image.Image:
    """Display the uploaded leaf image and return it as a PIL image."""
    image = Image.open(uploaded_file).convert("RGB")

    left_column, right_column = st.columns([1, 1])
    with left_column:
        st.subheader("Uploaded Image")
        st.image(image, caption=uploaded_file.name, use_container_width=True)

    with right_column:
        st.subheader("Image Details")
        st.write(f"**File name:** {uploaded_file.name}")
        st.write(f"**Image size:** {image.width} x {image.height}")
        st.write(f"**Image mode:** {image.mode}")

    return image


def render_prediction_result(prediction: PredictionResult) -> None:
    """Display disease prediction, confidence, and recommendation details."""
    st.divider()
    st.subheader("Prediction Result")

    result_column, details_column = st.columns([1, 2])

    with result_column:
        st.metric("Predicted Class", prediction.disease_name)
        st.metric("Confidence", f"{prediction.confidence:.1f}%")
        st.metric("Severity", prediction.severity)
        st.progress(int(prediction.confidence))

    with details_column:
        st.write("**Description**")
        st.write(prediction.description)
        st.write("**Recommended Action**")
        st.write(prediction.recommendation)
