"""
Aerial Object Classification & Detection
-----------------------------------------
Streamlit Deployment App
Run:  streamlit run C:\VSCODE\Aerial_project\streamlit_app.py
"""

import os
import io
import numpy as np
from pathlib import Path
from PIL import Image
import streamlit as st


# ─────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Aerial Object Classifier",
    page_icon="🛸",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# Styling
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        color: #1E3A5F;
        text-align: center;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #555;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .result-box {
        padding: 1rem 1.5rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    .bird-box  { background: #e8f5e9; border-left: 5px solid #2e7d32; }
    .drone-box { background: #fce4ec; border-left: 5px solid #c62828; }
    .metric-label { font-weight: 600; color: #333; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# Model loading (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading classification model …")
def load_classifier(model_path: str):
    import tensorflow as tf
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)


# ─────────────────────────────────────────────────────────────
# Inference helpers
# ─────────────────────────────────────────────────────────────
CLASS_LABELS = {0: '🐦 Bird', 1: '🚁 Drone'}
CLASS_COLORS = {0: 'bird-box', 1: 'drone-box'}
IMG_SIZE = (224, 224)


def preprocess_for_classifier(pil_img, preprocess_fn=None):
    img = pil_img.convert('RGB').resize(IMG_SIZE)
    arr = np.array(img, dtype=np.float32)
    if preprocess_fn:
        arr = preprocess_fn(arr)
    else:
        arr = arr / 255.0
    return np.expand_dims(arr, axis=0)


def classify(model, pil_img, preprocess_fn=None):
    """Return (label_str, confidence, raw_prob)."""
    x = preprocess_for_classifier(pil_img, preprocess_fn)
    prob = float(model.predict(x, verbose=0)[0][0])
    pred_class = 1 if prob >= 0.5 else 0
    confidence = prob if pred_class == 1 else 1 - prob
    return CLASS_LABELS[pred_class], confidence, prob


# ─────────────────────────────────────────────────────────────
# Sidebar – settings
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/drone.png", width=80)
    st.title("⚙️ Settings")

    st.subheader("Classification Model")
    # UPDATED: Replaced default with the absolute path provided
    clf_path = st.text_input(
        "Model path (.keras / .h5)",
        value=r"C:\VSCODE\Aerial_project\best_model.keras",
    )
    use_preprocess = st.selectbox(
        "Preprocessing",
        ['EfficientNet', 'MobileNetV2', 'ResNet50', 'None (÷255)'],
    )

    st.divider()
    st.caption("Aerial Object Classifier v1.0")


# ─────────────────────────────────────────────────────────────
# Load models
# ─────────────────────────────────────────────────────────────
def get_preprocess_fn(name: str):
    import tensorflow as tf
    mapping = {
        'EfficientNet':
            tf.keras.applications.efficientnet.preprocess_input,
        'MobileNetV2':
            tf.keras.applications.mobilenet_v2.preprocess_input,
        'ResNet50':
            tf.keras.applications.resnet50.preprocess_input,
        'None (÷255)': None,
    }
    return mapping.get(name)

# ADDED: Instantiate the variables so they exist for the main UI to use
preprocess_fn = get_preprocess_fn(use_preprocess)
clf_model = load_classifier(clf_path)


# ─────────────────────────────────────────────────────────────
# Main UI
# ─────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🛸 Aerial Object Classifier</div>',
            unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Upload an aerial image to classify '
    'as <strong>Bird</strong> or <strong>Drone</strong></div>',
    unsafe_allow_html=True,
)
st.divider()

# Upload
col1, col2 = st.columns([1, 1])
with col1:
    uploaded = st.file_uploader(
        "Upload an image (JPG / PNG)",
        type=['jpg', 'jpeg', 'png'],
    )
    use_sample = st.checkbox("Use a sample image")

if use_sample:
    # Placeholder sample – replace with actual sample paths
    st.info("Place sample images in `samples/bird.jpg` "
            "and `samples/drone.jpg`")

if uploaded is not None:
    pil_img = Image.open(uploaded)

    with col1:
        st.subheader("📷 Uploaded Image")
        st.image(pil_img, use_column_width=True)

    with col2:
        st.subheader("🔍 Results")

        # ── Classification ───────────────────────────────────
        if clf_model is None:
            st.warning(f"Classification model not found at:\n`{clf_path}`")
        else:
            with st.spinner("Classifying …"):
                label, conf, raw_prob = classify(
                    clf_model, pil_img, preprocess_fn
                )
            box_cls = 'bird-box' if 'Bird' in label else 'drone-box'
            st.markdown(
                f'<div class="result-box {box_cls}">'
                f'<span class="metric-label">Prediction:</span> '
                f'<strong style="font-size:1.4rem">{label}</strong><br>'
                f'<span class="metric-label">Confidence:</span> '
                f'{conf*100:.1f}%'
                f'</div>',
                unsafe_allow_html=True,
            )

            # Probability bar
            st.subheader("Probability Distribution")
            prob_bird  = 1 - raw_prob
            prob_drone = raw_prob
            st.progress(int(prob_bird * 100),
                        text=f"🐦 Bird  {prob_bird*100:.1f}%")
            st.progress(int(prob_drone * 100),
                        text=f"🚁 Drone {prob_drone*100:.1f}%")


# ─────────────────────────────────────────────────────────────
# About tab at bottom
# ─────────────────────────────────────────────────────────────
st.divider()
with st.expander("ℹ️ About this App"):
    st.markdown("""
    ### Aerial Object Classification & Detection

    This application uses deep learning to distinguish **birds** from
    **drones** in aerial imagery.

    **Models used:**
    - Custom CNN (trained from scratch)
    - Transfer Learning (EfficientNetB0 / MobileNetV2 / ResNet50)
    - YOLOv8 (optional object detection with bounding boxes)

    **Use cases:**
    - Airport bird-strike prevention
    - Restricted airspace drone detection
    - Wildlife monitoring

    **Dataset:** Binary classification (Bird / Drone) with ~3,300 images.
    """)
