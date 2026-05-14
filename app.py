"""Streamlit entry point for the Plant Disease Prediction app."""

import streamlit as st

from src.config import APP_TITLE, APP_SUBTITLE, SUPPORTED_IMAGE_TYPES
from src.services.demo_prediction_service import predict_leaf_disease_demo
from src.ui.layout import configure_page, render_header, render_sidebar
from src.ui.prediction_view import render_prediction_result, render_uploaded_image
from src.ui.session_state import add_prediction_to_history, initialize_session_state


def main() -> None:
    """Run the Streamlit application."""
    configure_page()
    initialize_session_state()

    render_sidebar()
    render_header(APP_TITLE, APP_SUBTITLE)

    uploaded_file = st.file_uploader(
        "Upload a plant leaf image",
        type=SUPPORTED_IMAGE_TYPES,
        help="Use a clear image of one leaf. JPG, JPEG, and PNG are supported.",
    )

    if uploaded_file is None:
        st.info("Upload a leaf image to start the prediction demo.")
        return

    image = render_uploaded_image(uploaded_file)

    if st.button("Analyze Leaf", type="primary", use_container_width=True):
        with st.spinner("Analyzing the leaf image..."):
            prediction = predict_leaf_disease_demo(uploaded_file.name)
            add_prediction_to_history(prediction)

        render_prediction_result(prediction)

    if st.session_state.prediction_history:
        st.divider()
        st.subheader("Prediction History")
        for item in reversed(st.session_state.prediction_history[-5:]):
            st.write(f"**{item.disease_name}** — {item.confidence:.1f}% confidence")


if __name__ == "__main__":
    main()
