import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import pickle
import cv2
import os
import matplotlib.pyplot as plt

# -----------------------------------------------------------
# ✅ CONFIGURATION
# -----------------------------------------------------------
MODEL_PATH = r"C:\Users\Jenifer\Documents\Crater\crater_detection_model.keras"
CLASS_MAP_PKL = r"C:\Users\Jenifer\Documents\Crater\class_map.pkl"
PATCH_SIZE = 32  # same as model input size
STEP_SIZE = 16   # sliding window stride
THRESHOLD = 0.8  # confidence threshold

# -----------------------------------------------------------
# ✅ HELPER FUNCTIONS
# -----------------------------------------------------------
@st.cache_resource
def load_model_and_map():
    if not os.path.exists(MODEL_PATH):
        st.error(f" Model not found at: {MODEL_PATH}")
        st.stop()
    model = tf.keras.models.load_model(MODEL_PATH)
    if os.path.exists(CLASS_MAP_PKL):
        with open(CLASS_MAP_PKL, "rb") as f:
            class_map = pickle.load(f)
    else:
        class_map = {0: "no_crater", 1: "crater"}
    return model, class_map


def slide_and_detect(img_gray, model):
    """Slide window across image and detect craters."""
    h, w = img_gray.shape
    detections = []

    for y in range(0, h - PATCH_SIZE, STEP_SIZE):
        for x in range(0, w - PATCH_SIZE, STEP_SIZE):
            patch = img_gray[y:y+PATCH_SIZE, x:x+PATCH_SIZE]
            patch_norm = patch.astype("float32") / 255.0
            patch_input = np.expand_dims(patch_norm, axis=(0, -1))
            pred = model.predict(patch_input, verbose=0)[0][0]
            if pred >= THRESHOLD:
                detections.append((x + PATCH_SIZE//2, y + PATCH_SIZE//2, pred))
    return detections


def draw_detections(img_rgb, detections):
    """Draw red circles where craters detected."""
    overlay = img_rgb.copy()
    for (cx, cy, conf) in detections:
        cv2.circle(overlay, (cx, cy), 10, (255, 0, 0), 2)
    return overlay

# -----------------------------------------------------------
# ✅ STREAMLIT APP
# -----------------------------------------------------------
st.set_page_config(page_title=" Lunar Crater Detection", layout="wide", page_icon="🌕")

st.title(" Lunar Crater Detection with Localization")
st.markdown("""
Upload a **Digital Elevation Map (DEM)** to visualize **where craters are detected**.
The model scans 32×32 patches across the image.
""")

uploaded = st.file_uploader(" Upload DEM image", type=["png", "jpg", "jpeg", "tif"])

if uploaded is None:
    st.info(" Please upload an image to begin crater localization.")
    st.stop()

# Load and convert to grayscale
img = Image.open(uploaded).convert("RGB")
img_gray = np.array(img.convert("L"))
st.image(img, caption=" Uploaded Image", use_container_width=True)

# Load model
with st.spinner(" Loading model and scanning for craters..."):
    model, class_map = load_model_and_map()
    detections = slide_and_detect(img_gray, model)

# Draw detection marks
result_img = draw_detections(np.array(img), detections)

# -----------------------------------------------------------
# ✅ DISPLAY RESULTS
# -----------------------------------------------------------
st.subheader(" Crater Localization Result")
st.image(result_img, caption=f"🪐 Detected {len(detections)} crater regions", use_container_width=True)

if len(detections) > 0:
    st.success(f" {len(detections)} potential crater points detected.")
    st.dataframe(
        [{"x": x, "y": y, "confidence": round(c, 3)} for (x, y, c) in detections],
        use_container_width=True
    )
else:
    st.warning("No craters detected above the threshold.")

st.caption("Developed by Jenifer • AI-Powered Lunar Crater Localization")
# streamlit run app.py
