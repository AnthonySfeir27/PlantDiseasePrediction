"""Page layout and global visual sections."""

import streamlit as st


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
        "This interface is prepared for a CNN transfer-learning model. "
        "The current version uses a demo prediction service until the model is trained."
    )


def render_sidebar() -> None:
    """Render project details in the sidebar."""
    with st.sidebar:
        st.header("Project Summary")
        st.write("**Course topics used:**")
        st.write("CNNs, image classification, transfer learning, preprocessing, confidence scoring")

        st.divider()
        st.write("**Current stage:** GUI prototype")
        st.write("**Next stage:** TensorFlow model training")

        st.divider()
        st.write("**Supported classes:**")
        st.write("Tomato Healthy")
        st.write("Tomato Early Blight")
        st.write("Tomato Late Blight")
        st.write("Tomato Leaf Mold")
