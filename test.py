import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import pickle
from pathlib import Path

# Load model and class map
MODEL_PATH = r"C:\Users\Jenifer\Documents\Crater\class_map.pkl\crater_detection_model.keras"
CLASS_MAP_PKL = r"C:\Users\Jenifer\Documents\Crater\class_map.pkl"
TEST_IMG_PATH = r"C:\Users\Jenifer\Documents\Crater\6.png"  # <-- your DEM in PNG

model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_MAP_PKL, "rb") as f:
    class_map = pickle.load(f)

print("✅ Model and class map loaded successfully!")

# --- Preprocess test image ---
img = cv2.imread(TEST_IMG_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("Test image not found!")

img = cv2.resize(img, (32, 32))  # match model input size
img_norm = img.astype(np.float32) / 255.0
input_data = np.expand_dims(np.expand_dims(img_norm, axis=0), axis=-1)

# --- Predict ---
pred = model.predict(input_data)[0][0]
label = 1 if pred >= 0.5 else 0
print(f"🛰️ Prediction: {class_map[label]} (Confidence: {pred:.2f})")

# --- Optional Grad-CAM visualization ---
from tensorflow.keras import models
from tensorflow.keras import backend as K

def make_gradcam_heatmap(img_array, model, last_conv_layer_name='last_conv'):
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, preds = grad_model(img_array)
        class_channel = preds[:, 0]
    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_outputs), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    heatmap /= np.max(heatmap)
    return cv2.resize(heatmap.numpy(), (32, 32))

# Compute Grad-CAM
heatmap = make_gradcam_heatmap(input_data, model)
heatmap_colored = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
overlay = cv2.addWeighted(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), 0.6, heatmap_colored, 0.4, 0)

# --- Show ---
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Input DEM")
plt.axis("off")

plt.subplot(1, 2, 2)
plt.imshow(overlay[..., ::-1])
plt.title(f"Grad-CAM: {class_map[label]} ({pred:.2f})")
plt.axis("off")

plt.show()
