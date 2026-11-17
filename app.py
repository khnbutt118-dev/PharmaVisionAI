from datetime import datetime
from io import BytesIO
import hashlib

import pandas as pd
import streamlit as st
from PIL import Image

from src.modules.tablet_inspection import TabletInspector


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="PharmaSight AI",
    page_icon="💊",
    layout="wide"
)

st.title("💊 PharmaSight AI")
st.caption("AI-powered pharmaceutical tablet inspection research prototype")

st.warning(
    "Research prototype only. Predictions are not validated GMP release "
    "decisions and must not replace qualified pharmaceutical inspection."
)


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_inspector():
    return TabletInspector()


try:
    inspector = load_inspector()
    st.sidebar.success("Three-class AI model loaded")
    st.sidebar.write("Device:", str(inspector.device))
except Exception as error:
    st.error(f"Could not load the inspection model: {error}")
    st.stop()


# --------------------------------------------------
# SESSION HISTORY
# --------------------------------------------------

if "inspection_history" not in st.session_state:
    st.session_state.inspection_history = []

if "last_inspected_hash" not in st.session_state:
    st.session_state.last_inspected_hash = None


# --------------------------------------------------
# INSPECTION DISPLAY
# --------------------------------------------------

def display_inspection(image, source):
    image_bytes = BytesIO()
    image.save(image_bytes, format="PNG")
    image_data = image_bytes.getvalue()

    image_hash = hashlib.sha256(image_data).hexdigest()

    st.image(image, caption="Tablet image", width=400)

    if image_hash != st.session_state.last_inspected_hash:

        with st.spinner("Inspecting tablet..."):
            result = inspector.inspect_image(image)

        st.session_state.last_inspected_hash = image_hash

        st.session_state.inspection_history.insert(
            0,
            {
                "Time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "Source": source,
                "Prediction": result["predicted_class"],
                "Confidence": round(result["confidence"] * 100, 2),
                "Image": image_hash[:12]
            }
        )

    else:
        result = inspector.inspect_image(image)

    st.divider()
    st.subheader("Inspection result")

    predicted = result["predicted_class"]
    confidence = result["confidence"] * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Predicted class", predicted.title())

    with col2:
        st.metric("Model confidence", f"{confidence:.2f}%")

    if result["review_required"]:
        st.warning(
            "Low model confidence. Send this image for manual review."
        )
    elif predicted == "normal":
        st.success("Model prediction: Normal tablet")
    else:
        st.error(f"Model prediction: {predicted.title()}")

    st.subheader("Class probabilities")

    for class_name, probability in result["probabilities"].items():
        st.write(f"**{class_name.title()}** — {probability * 100:.2f}%")
        st.progress(probability)

    st.caption(
        "These scores are model outputs, not calibrated probabilities "
        "or proof that the tablet is defect-free."
    )


# --------------------------------------------------
# MAIN INSPECTION INTERFACE
# --------------------------------------------------

st.header("Tablet inspection")

upload_tab, camera_tab = st.tabs(
    ["📁 Upload image", "📷 Camera capture"]
)

with upload_tab:
    st.write("Upload a photograph of one tablet.")

    uploaded_file = st.file_uploader(
        "Choose tablet image",
        type=["jpg", "jpeg", "png", "bmp"]
    )

    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
            display_inspection(image, "Image upload")
        except Exception as error:
            st.error(f"Could not process this image: {error}")

with camera_tab:
    st.write("Position one tablet clearly in front of your webcam.")

    camera_image = st.camera_input("Capture tablet")

    if camera_image is not None:
        try:
            image = Image.open(camera_image).convert("RGB")
            display_inspection(image, "Webcam")
        except Exception as error:
            st.error(f"Could not process camera image: {error}")


# --------------------------------------------------
# INSPECTION HISTORY
# --------------------------------------------------

st.divider()
st.header("Inspection history")

history = st.session_state.inspection_history

if history:
    history_df = pd.DataFrame(history)

    st.dataframe(
        history_df.drop(columns=["Image"]),
        use_container_width=True,
        hide_index=True
    )

    csv_data = history_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download inspection history (CSV)",
        data=csv_data,
        file_name="pharmasight_inspection_history.csv",
        mime="text/csv"
    )

    if st.button("Clear session history"):
        st.session_state.inspection_history = []
        st.session_state.last_inspected_hash = None
        st.rerun()
else:
    st.info("No tablets inspected in this session yet.")


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()
st.caption(
    "PharmaSight AI | Research prototype | "
    "Normal • Contamination • Chipping"
)
