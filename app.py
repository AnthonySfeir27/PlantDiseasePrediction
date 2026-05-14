"""Streamlit entry point for the Plant Disease Prediction app."""

import streamlit as st

from src.config import APP_SUBTITLE, APP_TITLE, SUPPORTED_IMAGE_TYPES
from src.services.demo_prediction_service import predict_leaf_disease_demo
from src.services.prediction_service import get_model_status, predict_leaf_disease
from src.ui.layout import configure_page, render_header, render_sidebar
from src.ui.prediction_view import render_prediction_result, render_training_outputs, render_uploaded_image
from src.ui.session_state import add_prediction_to_history, initialize_session_state


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    initialize_session_state()

    model_status = get_model_status()
    render_sidebar(model_status)
    render_header(APP_TITLE, APP_SUBTITLE)
    render_training_outputs()

    uploaded_file = st.file_uploader(
        "Upload a tomato leaf image",
        type=SUPPORTED_IMAGE_TYPES,
        help="Use a clear image of one tomato leaf. JPG, JPEG, and PNG are supported.",
    )

    if uploaded_file is None:
        st.info("Upload a leaf image to start analysis.")
        return

    image = render_uploaded_image(uploaded_file)

    if st.button("Analyze Leaf", type="primary", width="stretch"):
        with st.spinner("Analyzing the leaf image..."):
            if model_status.is_ready:
                prediction = predict_leaf_disease(image)
            else:
                prediction = predict_leaf_disease_demo(image, uploaded_file.name)
            add_prediction_to_history(prediction)

        render_prediction_result(prediction)

    render_prediction_history()


def render_prediction_history() -> None:
    """Render the latest predictions from the current session."""
    if not st.session_state.prediction_history:
        return

    st.divider()
    st.subheader("Prediction History")
    for item in reversed(st.session_state.prediction_history[-5:]):
        source = "real model" if item.is_real_model else "demo"
        st.write(f"**{item.disease_name}** — {item.confidence:.1f}% confidence ({source})")


if __name__ == "__main__":
    main()
