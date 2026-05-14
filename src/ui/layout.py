"""Page layout and global visual sections."""

import streamlit as st

from src.config import DATASET_DIR, MODEL_PATH, REQUIRED_DATASET_CLASSES
from src.models.model_status import ModelStatus


def configure_page() -> None:
    """Configure the Streamlit page before rendering widgets."""
    st.set_page_config(
        page_title="Plant Disease Prediction",
        page_icon="🌿",
        layout="wide",
        initial_sidebar_state="expanded",
    )


def render_header(title: str, subtitle: str) -> None:
    """Render the main app header."""
    st.title(title)
    st.caption(subtitle)
    st.markdown(
        "Upload a tomato leaf image and classify it using a CNN transfer-learning pipeline. "
        "The app uses a demo fallback until a trained TensorFlow model is available."
    )


def render_sidebar(model_status: ModelStatus) -> None:
    """Render project details in the sidebar."""
    with st.sidebar:
        st.header("Project Summary")
        st.write("**Course topics used:**")
        st.write("CNNs, image classification, transfer learning, data augmentation, confidence scoring")

        st.divider()
        st.write("**Model status:**")
        if model_status.is_ready:
            st.success(model_status.message)
        else:
            st.warning(model_status.message)

        st.write("**Model path:**")
        st.code(str(MODEL_PATH), language="text")

        st.divider()
        st.write("**Dataset folder:**")
        st.code(str(DATASET_DIR), language="text")

        st.write("**Required classes:**")
        for class_name in REQUIRED_DATASET_CLASSES:
            st.write(f"- {class_name}")

        st.divider()
        st.write("**Commands:**")
        st.code("python scripts/check_dataset.py", language="bash")
        st.code("python train.py --epochs 5", language="bash")
